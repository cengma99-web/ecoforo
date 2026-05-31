import pytest
from click.testing import CliRunner
from ecoforo.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "collect" in result.output
    assert "query" in result.output
    assert "backfill" in result.output
    assert "status" in result.output


def test_collect_no_args_shows_error(runner):
    result = runner.invoke(cli, ["collect"])
    assert result.exit_code != 0


def test_collect_dry_run(runner):
    result = runner.invoke(cli, [
        "collect", "--source", "metals", "--dry-run",
        "--start", "2026-01-01", "--end", "2026-01-02"
    ])
    # Should not crash even if yfinance returns empty
    assert result.exit_code == 0


def test_status(runner):
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0


def test_query_empty(runner):
    result = runner.invoke(cli, ["query"])
    assert result.exit_code == 0
