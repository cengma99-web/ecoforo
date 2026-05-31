"""Base fetcher abstract class for ecoforo economic data pipeline."""

import hashlib
import logging
import time
import traceback as tb_module
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone

from ecoforo.db.models import EconomicEvent, EventSource, DeadLetter

logger = logging.getLogger(__name__)


class FetcherError(Exception):
    """Non-retryable fetcher error."""

    pass


class BaseFetcher(ABC):
    """Abstract base class for all economic data fetchers.

    Subclasses must implement:
        fetch(start, end) -> list[dict]
        normalize(raw) -> dict

    And set class attributes:
        source_name: str
        source_type: str  ("calendar" | "indicator" | "news")
    """

    # Must be set by subclasses
    source_name: str
    source_type: str  # "calendar" | "indicator" | "news"

    # Retry configuration — can be overridden per fetcher
    max_retries: int = 3
    retry_delays: tuple = (1, 5, 25)

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(self, start: date, end: date) -> list[dict]:
        """Fetch raw records from the data source.

        Args:
            start: Start date (inclusive).
            end:   End date (inclusive).

        Returns:
            List of raw record dicts.
        """
        ...

    @abstractmethod
    def normalize(self, raw: dict) -> dict:
        """Convert one raw record into the standardized dict.

        Required keys:  source_name, title, event_date, country
        Optional keys:  description, actual_value, forecast_value,
                        previous_value, impact, importance, url, raw_data

        Args:
            raw: A single raw record from ``fetch()``.

        Returns:
            Standardized record dict.
        """
        ...

    # ------------------------------------------------------------------
    # Optional-override: validation
    # ------------------------------------------------------------------

    def validate(self, record: dict) -> bool:
        """Validate a normalized record.

        Default checks:
        - ``title`` is non-empty.
        - ``event_date`` is a valid ISO-8601 date / datetime.
        - ``country`` is non-empty.

        Override in subclasses for source-specific rules.

        Args:
            record: Normalized record dict.

        Returns:
            ``True`` if the record passes validation.
        """
        # -- title --
        title = record.get("title")
        if not title or not str(title).strip():
            logger.debug("Validation failed: empty title")
            return False

        # -- country --
        country = record.get("country")
        if not country or not str(country).strip():
            logger.debug("Validation failed: empty country")
            return False

        # -- event_date must be parseable --
        event_date = record.get("event_date")
        if not event_date:
            logger.debug("Validation failed: missing event_date")
            return False

        try:
            self._parse_event_date(event_date)
        except (ValueError, TypeError) as e:
            logger.debug(
                "Validation failed: invalid event_date %r: %s", event_date, e
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_event_date(value):
        """Parse *value* into a timezone-aware ``datetime``.

        Handles ``datetime``, ``date``, and ISO-8601 strings.
        Naive inputs are assumed to be UTC.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)

        # String
        s = str(value).strip()
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # Fallback: plain date "YYYY-MM-DD"
            dt = datetime.strptime(s, "%Y-%m-%d")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _make_dedup_key(self, record: dict) -> str:
        """Generate a deterministic SHA-256 dedup key.

        The key is built from::

            title | source_name | event_date | country

        Returns:
            64-character lowercase hex digest.
        """
        title = str(record.get("title", "")).strip()
        source = str(record.get("source_name", "")).strip()
        event_date = str(record.get("event_date", "")).strip()
        country = str(record.get("country", "")).strip()

        raw = f"{title}|{source}|{event_date}|{country}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_or_create_source(self, db):
        """Return the ``EventSource`` row for this fetcher, creating it if needed.

        Args:
            db: SQLAlchemy session.

        Returns:
            ``EventSource`` instance.
        """
        source = (
            db.query(EventSource)
            .filter(EventSource.name == self.source_name)
            .first()
        )
        if source is None:
            source = EventSource(
                name=self.source_name,
                source_type=self.source_type,
            )
            db.add(source)
            db.flush()  # assign an id without committing
            logger.info("Created new EventSource: %s", self.source_name)
        return source

    def _upsert_event(self, db, record: dict, dedup_key: str) -> bool:
        """Insert or update an ``EconomicEvent`` identified by *dedup_key*.

        Args:
            db:        SQLAlchemy session.
            record:    Normalized record dict.
            dedup_key: SHA-256 hex dedup key.

        Returns:
            ``True`` if a new row was inserted, ``False`` if an existing row
            was updated.
        """
        source = self._get_or_create_source(db)

        existing = (
            db.query(EconomicEvent)
            .filter(EconomicEvent.dedup_key == dedup_key)
            .first()
        )

        event_date = self._parse_event_date(record["event_date"])

        if existing is not None:
            # -- update --
            existing.title = record.get("title", existing.title)
            existing.description = record.get("description", existing.description)
            existing.event_date = event_date
            existing.actual_value = record.get("actual_value", existing.actual_value)
            existing.forecast_value = record.get(
                "forecast_value", existing.forecast_value
            )
            existing.previous_value = record.get(
                "previous_value", existing.previous_value
            )
            existing.impact = record.get("impact", existing.impact)
            existing.country = record.get("country", existing.country)
            existing.importance = record.get("importance", existing.importance)
            existing.url = record.get("url", existing.url)
            existing.raw_data = record.get("raw_data", existing.raw_data)
            existing.updated_at = datetime.now(timezone.utc)
            logger.debug("Updated existing event %s...", dedup_key[:12])
            return False

        # -- insert --
        event = EconomicEvent(
            source_id=source.id,
            title=record["title"],
            description=record.get("description"),
            event_date=event_date,
            actual_value=record.get("actual_value"),
            forecast_value=record.get("forecast_value"),
            previous_value=record.get("previous_value"),
            impact=record.get("impact"),
            country=record.get("country"),
            importance=record.get("importance", 3),
            url=record.get("url"),
            raw_data=record.get("raw_data"),
            dedup_key=dedup_key,
        )
        db.add(event)
        logger.debug("Created new event %s...", dedup_key[:12])
        return True

    def _record_dead_letter(
        self, db, raw_data: dict, error: str, tb: str = ""
    ) -> None:
        """Persist a failed record to the ``DeadLetter`` table.

        Args:
            db:       SQLAlchemy session.
            raw_data: The raw / partially-processed data.
            error:    Human-readable error message.
            tb:       Traceback string (optional).
        """
        dead = DeadLetter(
            source_name=self.source_name,
            raw_data=raw_data,
            error_message=error,
            traceback=tb,
        )
        db.add(dead)
        logger.warning(
            "Dead letter [%s]: %s", self.source_name, error[:120]
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        db,
        start: date,
        end: date,
        dry_run: bool = False,
    ) -> dict:
        """Execute the full fetch pipeline::

            fetch  ->  normalize  ->  validate  ->  upsert

        The *fetch* step is retried up to ``max_retries`` times with
        progressive delays from ``retry_delays``.

        Args:
            db:      SQLAlchemy session.
            start:   Start date (inclusive).
            end:     End date (inclusive).
            dry_run: If ``True``, validate + count but do **not** write to
                     the database.

        Returns:
            Counts dict with keys:
            ``fetched``, ``normalized``, ``valid``, ``inserted``,
            ``updated``, ``dead_letters``.
        """
        counts: dict[str, int] = {
            "fetched": 0,
            "normalized": 0,
            "valid": 0,
            "inserted": 0,
            "updated": 0,
            "dead_letters": 0,
        }

        # ---- fetch (with retry) ---------------------------------------
        raw_records: list[dict] = []
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                raw_records = self.fetch(start, end)
                counts["fetched"] = len(raw_records)
                logger.info(
                    "[%s] Fetched %d raw records (%s -> %s)",
                    self.source_name,
                    counts["fetched"],
                    start,
                    end,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = (
                        self.retry_delays[attempt]
                        if attempt < len(self.retry_delays)
                        else self.retry_delays[-1]
                    )
                    logger.warning(
                        "[%s] Fetch attempt %d/%d failed: %s. Retrying in %ds…",
                        self.source_name,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[%s] All %d fetch attempts exhausted.",
                        self.source_name,
                        self.max_retries + 1,
                    )
                    raise FetcherError(
                        f"Fetch failed after {self.max_retries + 1} attempts: "
                        f"{last_error}"
                    ) from last_error

        # ---- process each record --------------------------------------
        for raw in raw_records:
            counts["normalized"] += 1

            # -- normalize --
            try:
                record = self.normalize(raw)
            except Exception as exc:
                logger.warning("[%s] Normalization error: %s", self.source_name, exc)
                if not dry_run:
                    self._record_dead_letter(
                        db, raw, f"Normalization error: {exc}", tb_module.format_exc()
                    )
                counts["dead_letters"] += 1
                continue

            # -- validate --
            try:
                is_valid = self.validate(record)
            except Exception as exc:
                logger.warning("[%s] validate() raised: %s", self.source_name, exc)
                if not dry_run:
                    self._record_dead_letter(
                        db,
                        record.get("raw_data", raw),
                        f"Validation error: {exc}",
                        tb_module.format_exc(),
                    )
                counts["dead_letters"] += 1
                continue

            if not is_valid:
                logger.debug(
                    "[%s] Record failed validation: %.80s",
                    self.source_name,
                    record.get("title", "?"),
                )
                if not dry_run:
                    self._record_dead_letter(
                        db,
                        record.get("raw_data", raw),
                        f"Validation failed: {record.get('title', 'unknown')}",
                    )
                counts["dead_letters"] += 1
                continue

            counts["valid"] += 1

            if dry_run:
                continue

            # -- upsert --
            try:
                dedup_key = self._make_dedup_key(record)
                is_new = self._upsert_event(db, record, dedup_key)
                if is_new:
                    counts["inserted"] += 1
                else:
                    counts["updated"] += 1
            except Exception as exc:
                logger.error("[%s] Upsert error: %s", self.source_name, exc)
                if not dry_run:
                    self._record_dead_letter(
                        db,
                        record.get("raw_data", raw),
                        f"Upsert error: {exc}",
                        tb_module.format_exc(),
                    )
                counts["dead_letters"] += 1

        # ---- update source timestamp ----------------------------------
        if not dry_run and counts["fetched"] > 0:
            try:
                source = self._get_or_create_source(db)
                source.last_fetch_at = datetime.now(timezone.utc)
            except Exception as exc:
                logger.warning(
                    "[%s] Could not update last_fetch_at: %s",
                    self.source_name,
                    exc,
                )

        # ---- commit ---------------------------------------------------
        if not dry_run:
            try:
                db.commit()
            except Exception as exc:
                logger.error("[%s] Commit failed: %s", self.source_name, exc)
                db.rollback()
                raise FetcherError(f"Database commit failed: {exc}") from exc

        logger.info(
            "[%s] Run complete — fetched=%d valid=%d inserted=%d "
            "updated=%d dead_letters=%d",
            self.source_name,
            counts["fetched"],
            counts["valid"],
            counts["inserted"],
            counts["updated"],
            counts["dead_letters"],
        )

        return counts

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(self, db, start: date, end: date) -> dict:
        """Run the pipeline and record ``last_backfill_at`` on the source.

        This is identical to :meth:`run` except the source row's
        ``last_backfill_at`` column is also updated.

        Args:
            db:    SQLAlchemy session.
            start: Start date (inclusive).
            end:   End date (inclusive).

        Returns:
            Counts dict (same schema as :meth:`run`).
        """
        result = self.run(db, start, end, dry_run=False)

        try:
            source = self._get_or_create_source(db)
            source.last_backfill_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(
                "[%s] Backfill recorded — last_backfill_at = %s",
                self.source_name,
                source.last_backfill_at,
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed to update last_backfill_at: %s",
                self.source_name,
                exc,
            )
            db.rollback()

        return result
