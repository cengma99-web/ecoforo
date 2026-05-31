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
@click.option("--months", default=12, help="Months of history for copper analysis.")
@click.option("--impact", is_flag=True, help="Show event impact analysis.")
def analyze(months, impact):
    """Run comprehensive economic analysis."""
    if impact:
        from ecoforo.predict.impact import analyze_event_impact
        click.echo(analyze_event_impact())
    else:
        from ecoforo.predict.analyze import run_full_analysis
        click.echo(run_full_analysis())


@cli.command()
@click.argument("commodity", default="copper")
@click.option("--train", is_flag=True, help="Train model for this commodity.")
@click.option("--backtest", is_flag=True, help="Run backtest.")
def predict(commodity, train, backtest):
    """Predict commodity price direction (30-day). Default: copper."""
    from ecoforo.predict.multi_predict import (
        COMMODITY_CONFIGS, predict_commodity, train_commodity,
        DIRECTION_LABELS, SIGNAL_LABELS,
    )

    config = COMMODITY_CONFIGS.get(commodity)
    if not config:
        available = ", ".join(COMMODITY_CONFIGS.keys())
        click.echo(f"Unknown commodity '{commodity}'. Available: {available}", err=True)
        sys.exit(1)

    if train:
        click.echo(f"Training {config['name']} model...")
        result = train_commodity(commodity)
        click.echo(
            f"  {config['name']}: CV={result['cv_accuracy']:.1%} "
            f"(baseline {result['baseline']:.1%}, +{result['improvement']:+.1%})"
        )
        click.echo(f"  Samples: {result['n_samples']}, Model: {result['model_path']}")

    try:
        pred = predict_commodity(commodity)
    except FileNotFoundError:
        click.echo(f"No trained model for {commodity}. Run with --train first.", err=True)
        sys.exit(1)

    click.echo("═" * 50)
    click.echo(f"{config['emoji']} {config['name']} 30日预测")
    click.echo("═" * 50)
    click.echo(f"当前: ${pred['current_price']:.2f}/{config['unit']}")
    click.echo(f"信号: {pred['signal']} {pred['direction_label']}")
    click.echo(f"预测: {pred['predicted_return_pct']:+.1f}% → ${pred['predicted_price']:.2f}/{config['unit']}")
    probs = pred['probabilities']
    click.echo(f"概率: 涨 {probs['up']:.0%} | 跌 {probs['down']:.0%}")
    click.echo(f"CV准确率: {pred['cv_accuracy']:.1%}  (30日方向)")


@cli.command()
def classify():
    """Classify existing GDELT news events."""
    from ecoforo.processing.classifier import classify_events_db
    result = classify_events_db()
    click.echo(f"Classified {result['classified']}/{result['total']} GDELT events")


@cli.command()
@click.option("--output", "-o", is_flag=True, help="Save to file instead of printing.")
def report(output):
    """Generate daily economic brief."""
    from ecoforo.report import generate_daily_report, save_report

    content = generate_daily_report()
    if output:
        path = save_report(content)
        click.echo(f"Report saved: {path}")
    else:
        click.echo(content)


if __name__ == "__main__":
    cli()
