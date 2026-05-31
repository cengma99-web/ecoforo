"""ecoforo CLI — economic data pipeline control."""

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import click

from ecoforo.config import config
from ecoforo.db.session import SessionLocal

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ecoforo")


def _get_fetchers():
    """Lazy-import and return all registered fetchers."""
    from ecoforo.fetchers.fred import FREDFetcher
    from ecoforo.fetchers.worldbank import WBFetcher
    from ecoforo.fetchers.metals import MetalsFetcher
    from ecoforo.fetchers.gdelt import GDELTFetcher
    from ecoforo.fetchers.china_macro import ChinaMacroFetcher

    return [
        FREDFetcher(),
        WBFetcher(),
        MetalsFetcher(),
        GDELTFetcher(),
        ChinaMacroFetcher(),
    ]


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """ecoforo — Economic data pipeline: collect, query, analyze."""


@cli.command()
@click.option("--source", "-s", multiple=True, help="Data source (fred, worldbank, metals, investing). Repeatable.")
@click.option("--all", "all_sources", is_flag=True, help="Run all active sources.")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD), default: yesterday.")
@click.option("--end", default=None, help="End date (YYYY-MM-DD), default: today.")
@click.option("--dry-run", is_flag=True, help="Fetch but don't write to database.")
def collect(source, all_sources, start, end, dry_run):
    """Collect economic data from sources."""
    if not all_sources and not source:
        click.echo("Specify --source or --all. Use --help for options.", err=True)
        sys.exit(1)

    end_date = date.fromisoformat(end) if end else date.today()
    start_date = date.fromisoformat(start) if start else end_date - timedelta(days=1)

    all_fetchers = _get_fetchers()
    selected = [f for f in all_fetchers if all_sources or f.source_name in source]

    if not selected:
        click.echo(f"No fetchers matched. Available: {[f.source_name for f in all_fetchers]}", err=True)
        sys.exit(1)

    db = SessionLocal()
    try:
        for fetcher in selected:
            click.echo(f"[{fetcher.source_name}] collecting {start_date} → {end_date} ...")
            counts = fetcher.run(db, start_date, end_date, dry_run=dry_run)
            click.echo(
                f"  fetched={counts['fetched']} normalized={counts['normalized']} "
                f"valid={counts['valid']} inserted={counts.get('inserted', 0)} "
                f"updated={counts.get('updated', 0)} dead_letters={counts.get('dead_letters', 0)}"
            )
    finally:
        db.close()


@cli.command()
@click.option("--today", is_flag=True, help="Show today's events.")
@click.option("--week", is_flag=True, help="Show this week's events.")
@click.option("--source", "-s", multiple=True, help="Filter by source name.")
@click.option("--country", "-c", multiple=True, help="Filter by country code (e.g. US, CN).")
@click.option("--limit", default=50, help="Maximum events to show.")
def query(today, week, source, country, limit):
    """Query economic events."""
    from ecoforo.db.models import EconomicEvent, EventSource

    db = SessionLocal()
    try:
        q = db.query(EconomicEvent).join(EventSource)

        if today:
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            q = q.filter(EconomicEvent.event_date >= today_start)
        elif week:
            week_start = datetime.now(timezone.utc) - timedelta(days=7)
            q = q.filter(EconomicEvent.event_date >= week_start)

        if source:
            q = q.filter(EventSource.name.in_(source))
        if country:
            q = q.filter(EconomicEvent.country.in_([c.upper() for c in country]))

        events = q.order_by(EconomicEvent.event_date.desc()).limit(limit).all()

        if not events:
            click.echo("No events found.")
            return

        click.echo(f"{'Date':<12} {'Country':<8} {'Import':<6} {'Title':<60} {'Actual':<12} {'Forecast':<12}")
        click.echo("-" * 110)
        for e in events:
            click.echo(
                f"{str(e.event_date)[:10]:<12} "
                f"{e.country or '--':<8} "
                f"{'★' * (e.importance or 0):<6} "
                f"{e.title[:58]:<60} "
                f"{str(e.actual_value or '--'):<12} "
                f"{str(e.forecast_value or '--'):<12}"
            )
    finally:
        db.close()


@cli.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD), default: today.")
@click.option("--source", "-s", "source_name", required=True, help="Source (fred, worldbank, metals, investing).")
def backfill(start, end, source_name):
    """Backfill historical data for a source."""
    end_date = date.fromisoformat(end) if end else date.today()
    start_date = date.fromisoformat(start)

    all_fetchers = _get_fetchers()
    fetcher = next((f for f in all_fetchers if f.source_name == source_name), None)
    if not fetcher:
        click.echo(f"Unknown source '{source_name}'. Available: {[f.source_name for f in all_fetchers]}", err=True)
        sys.exit(1)

    db = SessionLocal()
    try:
        click.echo(f"[{fetcher.source_name}] backfilling {start_date} → {end_date} ...")
        counts = fetcher.backfill(db, start_date, end_date)
        click.echo(
            f"  fetched={counts['fetched']} inserted={counts.get('inserted', 0)} "
            f"updated={counts.get('updated', 0)} dead_letters={counts.get('dead_letters', 0)}"
        )
    finally:
        db.close()


@cli.command()
def status():
    """Show data source status and counts."""
    from ecoforo.db.models import EventSource, EconomicEvent, DeadLetter

    db = SessionLocal()
    try:
        sources = db.query(EventSource).order_by(EventSource.name).all()
        if not sources:
            click.echo("No data sources registered. Run 'ecoforo collect --all' first.")
            return

        click.echo(f"{'Source':<20} {'Type':<12} {'Events':<10} {'Last Fetch':<22} {'Dead Letters':<12}")
        click.echo("-" * 80)
        for src in sources:
            event_count = db.query(EconomicEvent).filter(EconomicEvent.source_id == src.id).count()
            dl_count = db.query(DeadLetter).filter(DeadLetter.source_name == src.name).count()
            last_fetch = str(src.last_fetch_at)[:19] if src.last_fetch_at else "never"
            click.echo(
                f"{src.name:<20} "
                f"{str(src.source_type):<12} "
                f"{event_count:<10} "
                f"{last_fetch:<22} "
                f"{dl_count:<12}"
            )
    finally:
        db.close()


@cli.command()
@click.option("--train", is_flag=True, help="Train models before predicting.")
@click.option("--backtest", is_flag=True, help="Run backtest and show metrics.")
def predict(train, backtest):
    """Predict copper price direction (30-day)."""
    from ecoforo.predict.features import build_features
    from ecoforo.predict.train import train_models, save_models, load_models
    from ecoforo.predict.predict import format_prediction, format_backtest, run_backtest, predict_latest

    if train or backtest:
        click.echo("Building features and training model...")
        X, y_cls, y_reg = build_features()
        clf, reg, metrics = train_models(X, y_cls, y_reg)
        save_models(clf, reg, metrics)

    if backtest:
        result = run_backtest()
        click.echo(format_backtest(result))
    else:
        try:
            result = predict_latest()
        except FileNotFoundError:
            click.echo("No trained model found. Run with --train first.", err=True)
            sys.exit(1)
        click.echo(format_prediction(result))


if __name__ == "__main__":
    cli()
