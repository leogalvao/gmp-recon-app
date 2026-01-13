#!/usr/bin/env python3
"""
Modular Calibration CLI for GMP Reconciliation App.

This script provides targeted calibration commands for specific configuration
changes, avoiding full system retraining for small parameter updates.

Usage:
    # Show what would be calibrated based on config changes
    python calibrate.py auto --dry-run

    # Auto-detect changes and run minimal calibration
    python calibrate.py auto

    # Calibrate specific targets
    python calibrate.py target fuzzy_matching suggestion_engine

    # Use a predefined profile
    python calibrate.py profile quick
    python calibrate.py profile mapping
    python calibrate.py profile forecasting

    # Show available targets and profiles
    python calibrate.py list

    # Create baseline snapshots
    python calibrate.py baseline

    # Check what changed since last calibration
    python calibrate.py diff
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.calibration import (
    CalibrationRegistry,
    CalibrationResult,
    Calibrator,
    ConfigChangeDetector,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def print_result(result: CalibrationResult, verbose: bool = False) -> None:
    """Print a single calibration result."""
    status_colors = {
        "completed": "green",
        "failed": "red",
        "skipped": "yellow",
        "pending": "blue",
        "running": "cyan",
    }
    status = result.status.value
    color = status_colors.get(status, "white")

    click.echo(
        f"  {result.target_name:<30} "
        f"{click.style(status.upper(), fg=color):<12} "
        f"{format_duration(result.duration_seconds):>8}"
    )

    if result.message and verbose:
        click.echo(f"    {result.message}")

    if result.details and verbose:
        for key, value in result.details.items():
            click.echo(f"    {key}: {value}")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """GMP Modular Calibration System.

    Provides targeted calibration for configuration changes without
    requiring full system retraining.

    \b
    Profiles:
      quick       - Parameter updates only (no model training)
      mapping     - All mapping-related calibrations
      forecasting - Forecasting models and parameters
      schedule    - Schedule-driven system
      full        - Complete system recalibration
    """
    pass


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be calibrated")
@click.option("--no-snapshot", is_flag=True, help="Don't update config snapshots")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def auto(dry_run: bool, no_snapshot: bool, verbose: bool, output_json: bool):
    """Auto-detect config changes and calibrate affected targets.

    Compares current configuration files against saved snapshots to
    identify what has changed, then runs only the necessary calibrations.

    Example:
        python calibrate.py auto --dry-run
        python calibrate.py auto
    """
    click.echo(click.style("GMP Auto-Calibration", fg="cyan", bold=True))

    calibrator = Calibrator()
    summary = calibrator.auto_calibrate(
        update_snapshots=not no_snapshot,
        dry_run=dry_run,
    )

    if output_json:
        click.echo(json.dumps(summary.to_dict(), indent=2))
        return

    if not summary.targets_run:
        click.echo(click.style("\nNo changes detected. System is calibrated.", fg="green"))
        return

    click.echo(f"\nTriggered by: {summary.triggered_by or 'Manual'}")
    click.echo(f"Targets: {len(summary.targets_run)}")
    click.echo("-" * 60)

    click.echo(f"\n{'Target':<30} {'Status':<12} {'Duration':>8}")
    click.echo("-" * 60)

    for result in summary.results:
        print_result(result, verbose)

    click.echo("-" * 60)
    click.echo(
        f"\nCompleted: {summary.successful_count}/{len(summary.results)} "
        f"in {format_duration(summary.total_duration_seconds)}"
    )

    if summary.failed_count > 0:
        click.echo(click.style(f"Failed: {summary.failed_count}", fg="red"))
        sys.exit(1)


@cli.command()
@click.argument("targets", nargs=-1, required=True)
@click.option("--dry-run", is_flag=True, help="Show what would be calibrated")
@click.option("--no-deps", is_flag=True, help="Don't resolve dependencies")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def target(
    targets: tuple[str, ...],
    dry_run: bool,
    no_deps: bool,
    verbose: bool,
    output_json: bool,
):
    """Calibrate specific targets.

    Calibrates the specified targets and their dependencies (unless --no-deps).

    Example:
        python calibrate.py target fuzzy_matching
        python calibrate.py target suggestion_engine gmp_divisions
        python calibrate.py target ml_forecasting --no-deps
    """
    click.echo(click.style("GMP Target Calibration", fg="cyan", bold=True))

    calibrator = Calibrator()
    summary = calibrator.calibrate(
        list(targets),
        resolve_dependencies=not no_deps,
        dry_run=dry_run,
    )

    if output_json:
        click.echo(json.dumps(summary.to_dict(), indent=2))
        return

    click.echo(f"\nRequested: {', '.join(targets)}")
    if not no_deps and len(summary.targets_run) > len(targets):
        click.echo(f"With dependencies: {', '.join(summary.targets_run)}")
    click.echo("-" * 60)

    click.echo(f"\n{'Target':<30} {'Status':<12} {'Duration':>8}")
    click.echo("-" * 60)

    for result in summary.results:
        print_result(result, verbose)

    click.echo("-" * 60)
    click.echo(
        f"\nCompleted: {summary.successful_count}/{len(summary.results)} "
        f"in {format_duration(summary.total_duration_seconds)}"
    )

    if summary.failed_count > 0:
        sys.exit(1)


@cli.command()
@click.argument("profile_name")
@click.option("--dry-run", is_flag=True, help="Show what would be calibrated")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def profile(
    profile_name: str,
    dry_run: bool,
    verbose: bool,
    output_json: bool,
):
    """Calibrate using a predefined profile.

    Available profiles:
    \b
      quick       - Fast parameter updates (no model training)
      mapping     - GMP divisions, fuzzy matching, suggestions
      forecasting - ML models and forecasting parameters
      schedule    - Schedule parsing and models
      full        - Complete system recalibration

    Example:
        python calibrate.py profile quick
        python calibrate.py profile mapping --dry-run
        python calibrate.py profile full
    """
    click.echo(click.style(f"GMP Profile Calibration: {profile_name}", fg="cyan", bold=True))

    calibrator = Calibrator()
    summary = calibrator.calibrate_profile(profile_name, dry_run=dry_run)

    if output_json:
        click.echo(json.dumps(summary.to_dict(), indent=2))
        return

    if not summary.targets_run:
        click.echo(click.style(f"\nProfile '{profile_name}' not found or has no targets.", fg="yellow"))
        return

    registry = CalibrationRegistry()
    profile_obj = registry.get_profile(profile_name)
    if profile_obj:
        click.echo(f"\nDescription: {profile_obj.description}")

    click.echo(f"Targets: {len(summary.targets_run)}")
    estimated = registry.estimate_time(summary.targets_run)
    click.echo(f"Estimated time: {format_duration(estimated)}")
    click.echo("-" * 60)

    click.echo(f"\n{'Target':<30} {'Status':<12} {'Duration':>8}")
    click.echo("-" * 60)

    for result in summary.results:
        print_result(result, verbose)

    click.echo("-" * 60)
    click.echo(
        f"\nCompleted: {summary.successful_count}/{len(summary.results)} "
        f"in {format_duration(summary.total_duration_seconds)}"
    )

    if summary.failed_count > 0:
        sys.exit(1)


@cli.command("list")
@click.option("--targets", "-t", is_flag=True, help="List calibration targets")
@click.option("--profiles", "-p", is_flag=True, help="List calibration profiles")
def list_items(targets: bool, profiles: bool):
    """List available calibration targets and profiles.

    Example:
        python calibrate.py list
        python calibrate.py list --targets
        python calibrate.py list --profiles
    """
    registry = CalibrationRegistry()

    # If neither flag set, show both
    if not targets and not profiles:
        targets = True
        profiles = True

    if targets:
        click.echo(click.style("\nCalibration Targets:", fg="cyan", bold=True))
        click.echo("-" * 80)
        click.echo(f"{'Name':<25} {'Type':<20} {'Time':>8} {'Data?':>6}")
        click.echo("-" * 80)

        for name, target in sorted(registry.get_all_targets().items()):
            data_req = "Yes" if target.requires_data else "No"
            click.echo(
                f"{name:<25} {target.calibration_type:<20} "
                f"{format_duration(target.estimated_time_seconds):>8} {data_req:>6}"
            )

    if profiles:
        click.echo(click.style("\nCalibration Profiles:", fg="cyan", bold=True))
        click.echo("-" * 80)

        for name, profile_obj in sorted(registry.get_all_profiles().items()):
            click.echo(f"\n  {click.style(name, bold=True)}")
            click.echo(f"    {profile_obj.description}")

            if profile_obj.targets == "all":
                click.echo("    Targets: all")
            else:
                click.echo(f"    Targets: {', '.join(profile_obj.targets)}")

            if profile_obj.include_types:
                click.echo(f"    Include types: {', '.join(profile_obj.include_types)}")
            if profile_obj.exclude_types:
                click.echo(f"    Exclude types: {', '.join(profile_obj.exclude_types)}")


@cli.command()
@click.option("--file", "-f", "config_file", help="Specific config file to baseline")
def baseline(config_file: str | None):
    """Create baseline snapshots of configuration files.

    Creates snapshots that will be used to detect future changes.

    Example:
        python calibrate.py baseline
        python calibrate.py baseline -f gmp_mapping_config.yaml
    """
    click.echo(click.style("Creating Configuration Baselines", fg="cyan", bold=True))

    detector = ConfigChangeDetector()
    detector.create_baseline(config_file)

    if config_file:
        click.echo(f"\nBaseline created for: {config_file}")
    else:
        click.echo("\nBaselines created for all configuration files")

    click.echo(click.style("Done!", fg="green"))


@cli.command()
@click.option("--file", "-f", "config_file", help="Specific config file to check")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed changes")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def diff(config_file: str | None, verbose: bool, output_json: bool):
    """Show configuration changes since last calibration.

    Compares current config files to their saved snapshots.

    Example:
        python calibrate.py diff
        python calibrate.py diff -f gmp_mapping_config.yaml
        python calibrate.py diff --verbose
    """
    click.echo(click.style("Configuration Changes", fg="cyan", bold=True))

    detector = ConfigChangeDetector()

    if config_file:
        try:
            report = detector.detect_changes(config_file)
            reports = {config_file: report}
        except FileNotFoundError:
            click.echo(click.style(f"Config file not found: {config_file}", fg="red"))
            return
    else:
        reports = detector.detect_all_changes()

    if output_json:
        output = {k: v.to_dict() for k, v in reports.items()}
        click.echo(json.dumps(output, indent=2))
        return

    total_changes = 0
    all_targets = set()

    for name, report in reports.items():
        click.echo(f"\n{click.style(name, bold=True)}")

        snapshot_info = detector.get_snapshot_info(name)
        if snapshot_info:
            click.echo(f"  Last snapshot: {snapshot_info['timestamp']}")

        if not report.has_changes:
            click.echo(click.style("  No changes detected", fg="green"))
            continue

        click.echo(f"  Changes: {len(report.changes)}")
        click.echo(f"  Sections: {', '.join(report.sections_changed)}")
        click.echo(f"  Calibration targets: {', '.join(report.calibration_targets)}")

        total_changes += len(report.changes)
        all_targets.update(report.calibration_targets)

        if verbose:
            click.echo("\n  Changed values:")
            for change in report.changes[:10]:  # Limit to first 10
                old_str = str(change.old_value)[:30] if change.old_value else "(none)"
                new_str = str(change.new_value)[:30] if change.new_value else "(none)"
                click.echo(f"    {change.section}")
                click.echo(f"      {change.change_type}: {old_str} -> {new_str}")

            if len(report.changes) > 10:
                click.echo(f"    ... and {len(report.changes) - 10} more changes")

    click.echo("\n" + "-" * 60)
    if total_changes > 0:
        click.echo(f"Total changes: {total_changes}")
        click.echo(f"Targets needing calibration: {', '.join(sorted(all_targets))}")
        click.echo(f"\nRun: python calibrate.py auto")
    else:
        click.echo(click.style("No changes detected. System is calibrated.", fg="green"))


@cli.command()
@click.argument("target_name")
def info(target_name: str):
    """Show detailed information about a calibration target.

    Example:
        python calibrate.py info fuzzy_matching
        python calibrate.py info ml_forecasting
    """
    registry = CalibrationRegistry()
    target = registry.get_target(target_name)

    if not target:
        click.echo(click.style(f"Target not found: {target_name}", fg="red"))
        click.echo("\nAvailable targets:")
        for name in sorted(registry.get_all_targets().keys()):
            click.echo(f"  - {name}")
        return

    click.echo(click.style(f"\nCalibration Target: {target.name}", fg="cyan", bold=True))
    click.echo("-" * 60)
    click.echo(f"Description:     {target.description}")
    click.echo(f"Type:            {target.calibration_type}")
    click.echo(f"Estimated time:  {format_duration(target.estimated_time_seconds)}")
    click.echo(f"Requires data:   {'Yes' if target.requires_data else 'No'}")
    click.echo(f"Lightweight:     {'Yes' if target.is_lightweight else 'No'}")

    if target.data_sources:
        click.echo(f"\nData sources:")
        for source in target.data_sources:
            click.echo(f"  - {source}")

    if target.affects:
        click.echo(f"\nAffects:")
        for affected in target.affects:
            click.echo(f"  - {affected}")

    deps = registry.get_dependencies(target_name)
    if deps:
        click.echo(f"\nTriggers (dependencies):")
        for dep in deps:
            click.echo(f"  - {dep}")


if __name__ == "__main__":
    cli()
