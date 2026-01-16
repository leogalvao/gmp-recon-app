"""
Cutover CLI Commands - Management commands for project cutover operations.

Provides command-line interface for:
- Single project cutover
- Batch cutover operations
- Rollback operations
- Status monitoring
- Feature flag management
"""
import click
import logging
from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session

from app.models import get_db, Project
from app.domain.services import (
    ProjectCutoverService,
    ProjectMigrationService,
    FeatureStoreService,
    ModelTrainingService,
    ForecastInferenceService,
    LeakagePreventionService,
)
from app.infrastructure.feature_flags import FeatureFlags, FeatureFlag

logger = logging.getLogger(__name__)


@click.group()
def cutover():
    """Cutover management commands."""
    pass


@cutover.command()
@click.argument('project_id', type=int)
@click.option('--force', is_flag=True, help='Skip validation checks')
@click.option('--dry-run', is_flag=True, help='Validate without executing')
def execute(project_id: int, force: bool, dry_run: bool):
    """Execute cutover for a single project."""
    click.echo(f"Starting cutover for project {project_id}...")

    db = next(get_db())
    service = ProjectCutoverService(db)

    # Validate first
    is_ready, issues = service.validate_cutover_readiness(project_id)

    if not is_ready and not force:
        click.echo(click.style("Cutover blocked - issues found:", fg='red'))
        for issue in issues:
            click.echo(f"  - {issue}")
        if not dry_run:
            click.echo("\nUse --force to skip validation checks")
        return

    if dry_run:
        click.echo(click.style("Dry run - validation passed", fg='green'))
        return

    # Execute cutover
    result = service.execute_cutover(project_id, force=force)

    if result.success:
        click.echo(click.style(f"✓ Cutover successful for {result.project_code}", fg='green'))
        click.echo(f"  Quality score: {result.quality_score:.2f}")
        click.echo(f"  Forecast divergence: {result.forecast_divergence:.1%}")
    else:
        click.echo(click.style(f"✗ Cutover failed for project {project_id}", fg='red'))
        for error in result.errors:
            click.echo(f"  - {error}")


@cutover.command()
@click.argument('project_id', type=int)
def rollback(project_id: int):
    """Rollback a project to legacy system."""
    click.echo(f"Rolling back project {project_id}...")

    db = next(get_db())
    service = ProjectCutoverService(db)

    success = service.rollback_cutover(project_id)

    if success:
        click.echo(click.style(f"✓ Rollback successful for project {project_id}", fg='green'))
    else:
        click.echo(click.style(f"✗ Rollback failed for project {project_id}", fg='red'))


@cutover.command()
@click.argument('project_id', type=int)
def status(project_id: int):
    """Check cutover status for a project."""
    db = next(get_db())
    service = ProjectCutoverService(db)

    status = service.get_cutover_status(project_id)

    if 'error' in status:
        click.echo(click.style(status['error'], fg='red'))
        return

    click.echo(f"\nProject: {status['project_code']} (ID: {status['project_id']})")
    click.echo(f"Multi-project enabled: {status['is_multi_project_enabled']}")
    click.echo(f"Ready for cutover: {status['is_ready_for_cutover']}")
    click.echo(f"Data quality score: {status['data_quality_score']:.2f if status['data_quality_score'] else 'N/A'}")

    if status['blocking_issues']:
        click.echo("\nBlocking issues:")
        for issue in status['blocking_issues']:
            click.echo(f"  - {issue}")


@cutover.command()
@click.option('--min-quality', default=0.8, help='Minimum quality score')
@click.option('--force', is_flag=True, help='Skip validation checks')
@click.option('--dry-run', is_flag=True, help='Validate without executing')
@click.option('--limit', type=int, help='Maximum projects to process')
def batch(min_quality: float, force: bool, dry_run: bool, limit: Optional[int]):
    """Execute batch cutover for all eligible projects."""
    click.echo("Starting batch cutover...")

    db = next(get_db())

    # Get eligible projects
    query = db.query(Project).filter(
        Project.is_training_eligible == True,
        Project.data_quality_score >= min_quality,
    )

    if limit:
        query = query.limit(limit)

    projects = query.all()
    click.echo(f"Found {len(projects)} eligible projects")

    service = ProjectCutoverService(db)
    results = {'success': 0, 'failed': 0, 'skipped': 0}

    for project in projects:
        # Check if already enabled
        if FeatureFlags.MULTI_PROJECT_FORECASTING.is_enabled(project.id):
            click.echo(f"  Skipping {project.code} - already enabled")
            results['skipped'] += 1
            continue

        click.echo(f"  Processing {project.code}...", nl=False)

        if dry_run:
            is_ready, _ = service.validate_cutover_readiness(project.id)
            if is_ready or force:
                click.echo(click.style(" [WOULD CUTOVER]", fg='yellow'))
            else:
                click.echo(click.style(" [BLOCKED]", fg='red'))
            continue

        result = service.execute_cutover(project.id, force=force)

        if result.success:
            click.echo(click.style(" ✓", fg='green'))
            results['success'] += 1
        else:
            click.echo(click.style(" ✗", fg='red'))
            results['failed'] += 1

    click.echo(f"\nBatch cutover complete:")
    click.echo(f"  Success: {results['success']}")
    click.echo(f"  Failed: {results['failed']}")
    click.echo(f"  Skipped: {results['skipped']}")


@cutover.command()
@click.option('--percentage', type=float, help='Set rollout percentage (0-100)')
@click.option('--enable-all', is_flag=True, help='Enable for all projects')
@click.option('--disable-all', is_flag=True, help='Disable for all projects')
def flags(percentage: Optional[float], enable_all: bool, disable_all: bool):
    """Manage feature flags for multi-project system."""
    if percentage is not None:
        FeatureFlags.set_rollout_percentage(percentage)
        click.echo(f"Set rollout percentage to {percentage}%")
    elif enable_all:
        FeatureFlags.enable_globally()
        click.echo("Enabled all multi-project features globally")
    elif disable_all:
        for flag_name in ['multi_project_forecasting', 'canonical_trade_mapping',
                          'feature_store', 'probabilistic_forecasts']:
            FeatureFlag(flag_name).disable()
        click.echo("Disabled all multi-project features")
    else:
        # Show current status
        click.echo("\nFeature Flag Status:")
        for flag_status in FeatureFlag.list_all():
            status_str = flag_status['strategy']
            if flag_status['strategy'] == 'percentage':
                status_str = f"{flag_status['percentage']}%"
            click.echo(f"  {flag_status['name']}: {status_str}")


@cutover.command()
def validate_leakage():
    """Run leakage prevention validation suite."""
    click.echo("Running leakage prevention validation...")

    db = next(get_db())
    service = LeakagePreventionService(db)

    results = service.run_full_validation_suite()

    passed = 0
    failed = 0

    for test_name, result in results.items():
        if result.is_valid:
            click.echo(click.style(f"  ✓ {test_name}", fg='green'))
            passed += 1
        else:
            click.echo(click.style(f"  ✗ {test_name}: {result.details}", fg='red'))
            failed += 1

    click.echo(f"\nValidation complete: {passed} passed, {failed} failed")


@cutover.command()
@click.option('--project-id', type=int, help='Specific project to migrate')
@click.option('--all', 'migrate_all', is_flag=True, help='Migrate all projects')
def migrate(project_id: Optional[int], migrate_all: bool):
    """Run data migration (Phase 2) for projects."""
    db = next(get_db())
    service = ProjectMigrationService(db)

    if project_id:
        click.echo(f"Migrating project {project_id}...")
        result = service.migrate_project(project_id)
        db.commit()

        if result.success:
            click.echo(click.style(f"✓ Migration successful", fg='green'))
            click.echo(f"  Trades mapped: {result.trades_mapped}")
            click.echo(f"  Auto-confirmed: {result.trades_auto_confirmed}")
            click.echo(f"  Need review: {result.trades_need_review}")
        else:
            click.echo(click.style(f"✗ Migration failed", fg='red'))
            for error in result.errors:
                click.echo(f"  - {error}")

    elif migrate_all:
        click.echo("Migrating all projects...")
        result = service.migrate_all_projects()

        click.echo(f"\nMigration complete:")
        click.echo(f"  Total: {result.total_projects}")
        click.echo(f"  Successful: {result.successful}")
        click.echo(f"  Failed: {result.failed}")
        click.echo(f"  Trades mapped: {result.total_trades_mapped}")

    else:
        click.echo("Specify --project-id or --all")


@cutover.command()
@click.option('--project-id', type=int, help='Specific project to backfill')
@click.option('--all', 'backfill_all', is_flag=True, help='Backfill all projects')
@click.option('--period-type', default='monthly', help='Period type: weekly or monthly')
def backfill(project_id: Optional[int], backfill_all: bool, period_type: str):
    """Backfill feature store for projects."""
    db = next(get_db())
    service = FeatureStoreService(db)

    if project_id:
        click.echo(f"Backfilling features for project {project_id}...")
        result = service.backfill_project_features(project_id, period_type)
        db.commit()

        if result.errors:
            click.echo(click.style(f"✗ Backfill failed", fg='red'))
            for error in result.errors:
                click.echo(f"  - {error}")
        else:
            click.echo(click.style(f"✓ Backfill successful", fg='green'))
            click.echo(f"  Periods created: {result.periods_created}")
            click.echo(f"  Trades covered: {result.trades_covered}")

    elif backfill_all:
        click.echo("Backfilling features for all projects...")
        results = service.backfill_all_projects(period_type)

        total_periods = sum(r.periods_created for r in results.values())
        errors = sum(1 for r in results.values() if r.errors)

        click.echo(f"\nBackfill complete:")
        click.echo(f"  Projects processed: {len(results)}")
        click.echo(f"  Total periods: {total_periods}")
        click.echo(f"  Errors: {errors}")

    else:
        click.echo("Specify --project-id or --all")


@cutover.command()
@click.option('--name', default='multi_project_forecaster', help='Model name')
def train(name: str):
    """Train the global ML model."""
    click.echo(f"Training global model '{name}'...")
    click.echo("This may take a while...")

    db = next(get_db())
    service = ModelTrainingService(db)

    result = service.train_global_model(model_name=name)

    if result.success:
        click.echo(click.style(f"\n✓ Training successful", fg='green'))
        click.echo(f"  Model ID: {result.model_id}")
        click.echo(f"  Version: {result.model_version}")
        click.echo(f"  Epochs: {result.epochs_trained}")
        click.echo(f"  Final train loss: {result.final_train_loss:.4f}")
        if result.final_val_loss:
            click.echo(f"  Final val loss: {result.final_val_loss:.4f}")
    else:
        click.echo(click.style(f"\n✗ Training failed", fg='red'))
        for error in result.errors:
            click.echo(f"  - {error}")


@cutover.command()
@click.argument('project_id', type=int)
def forecast(project_id: int):
    """Generate forecast for a project."""
    click.echo(f"Generating forecast for project {project_id}...")

    db = next(get_db())
    service = ForecastInferenceService(db)

    result = service.get_project_forecast(project_id)

    if not result:
        click.echo(click.style("No forecast generated - check project data", fg='red'))
        return

    click.echo(click.style(f"\nForecast for {result.project_code}", fg='green'))
    click.echo(f"  As of: {result.as_of_date}")
    click.echo(f"  Model: {result.model_version}")
    click.echo(f"\n  Total Budget:     ${result.total_budget:,.2f}")
    click.echo(f"  Current Cost:     ${result.total_cumulative_cost:,.2f}")
    click.echo(f"  Forecasted EAC:   ${result.total_forecasted_eac:,.2f}")
    click.echo(f"  Range ({result.confidence_level:.0%}):   ${result.total_forecast_lower:,.2f} - ${result.total_forecast_upper:,.2f}")

    if result.trade_forecasts:
        click.echo(f"\n  By Trade:")
        for tf in result.trade_forecasts[:5]:  # Show top 5
            click.echo(f"    {tf.canonical_code}: ${tf.forecasted_eac:,.2f}")


# Register with main CLI if exists
def register_commands(cli):
    """Register cutover commands with main CLI."""
    cli.add_command(cutover)
