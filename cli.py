#!/usr/bin/env python3
"""
CLI for GMP Forecasting System.

Usage:
    python cli.py train --config config/training_config.yaml --data-path data/costs.csv --building-data data/buildings.csv
    python cli.py predict --sqft 50000 --stories 3 --green-roof
    python cli.py serve --port 8000

Commands:
    train     Train a forecasting model on historical data
    predict   Generate a cost forecast for a building
    serve     Start the API server
    info      Display model information
"""
import click
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """GMP Cost Forecasting System CLI.

    Train machine learning models for construction cost prediction
    using building parameters and historical cost data.
    """
    pass


@cli.command()
@click.option(
    '--config',
    default='config/training_config.yaml',
    help='Path to training configuration YAML file',
    type=click.Path(exists=True)
)
@click.option(
    '--data-path',
    required=True,
    help='Path to historical costs CSV (columns: project_id, date, cost)',
    type=click.Path(exists=True)
)
@click.option(
    '--building-data',
    required=True,
    help='Path to building parameters CSV',
    type=click.Path(exists=True)
)
@click.option(
    '--output',
    default='models/model.keras',
    help='Output path for trained model'
)
@click.option(
    '--epochs',
    default=None,
    type=int,
    help='Override training epochs from config'
)
@click.option(
    '--batch-size',
    default=None,
    type=int,
    help='Override batch size from config'
)
def train(config: str, data_path: str, building_data: str, output: str,
          epochs: int, batch_size: int):
    """Train a forecasting model on historical cost data.

    Loads configuration from YAML, prepares training data, and trains
    either an LSTM or Transformer model based on configuration.

    Example:
        python cli.py train \\
            --config config/training_config.yaml \\
            --data-path data/historical_costs.csv \\
            --building-data data/buildings.csv \\
            --output models/production.keras
    """
    click.echo(click.style('GMP Cost Forecasting - Training', fg='cyan', bold=True))
    click.echo(f"Loading configuration from: {config}")

    from app.infrastructure.ml.training_pipeline import TrainingPipeline, TrainingConfig

    # Load config and optionally override
    pipeline_config = TrainingConfig.from_yaml(config)
    if epochs is not None:
        pipeline_config.epochs = epochs
    if batch_size is not None:
        pipeline_config.batch_size = batch_size

    pipeline = TrainingPipeline(config=pipeline_config)

    click.echo(f"Model architecture: {pipeline_config.architecture}")
    click.echo(f"Lookback months: {pipeline_config.lookback_months}")
    click.echo(f"Forecast horizon: {pipeline_config.forecast_horizon}")

    # Load data
    click.echo(f"\nLoading data from: {data_path}")
    historical_costs = pd.read_csv(data_path)
    click.echo(f"  - Cost records: {len(historical_costs):,}")

    click.echo(f"Loading building data from: {building_data}")
    building_df = pd.read_csv(building_data)
    click.echo(f"  - Buildings: {len(building_df):,}")

    # Train
    click.echo("\n" + "=" * 50)
    click.echo("Starting training...")
    click.echo("=" * 50 + "\n")

    try:
        history = pipeline.train(historical_costs, building_df)

        # Save model
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save(str(output_path))

        click.echo("\n" + "=" * 50)
        click.echo(click.style("Training Complete!", fg='green', bold=True))
        click.echo("=" * 50)
        click.echo(f"Final loss: {history['loss'][-1]:.6f}")
        if 'val_loss' in history:
            click.echo(f"Validation loss: {history['val_loss'][-1]:.6f}")
        click.echo(f"Model saved to: {output}")

    except Exception as e:
        click.echo(click.style(f"Training failed: {e}", fg='red'), err=True)
        raise click.Abort()


@cli.command()
@click.option(
    '--model',
    default='models/model.keras',
    help='Path to trained model',
    type=click.Path(exists=True)
)
@click.option('--sqft', type=float, required=True, help='Building square footage')
@click.option('--stories', type=int, required=True, help='Number of stories')
@click.option('--green-roof', is_flag=True, help='Has green roof')
@click.option('--rooftop-units', type=int, default=0, help='Number of rooftop HVAC units')
@click.option('--fall-anchors', type=int, default=0, help='Number of fall anchors')
@click.option(
    '--history',
    type=str,
    default=None,
    help='Comma-separated historical monthly costs (e.g., "150000,162000,148000")'
)
@click.option('--confidence', type=float, default=0.80, help='Confidence level (0-1)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def predict(model: str, sqft: float, stories: int, green_roof: bool,
            rooftop_units: int, fall_anchors: int, history: str,
            confidence: float, output_json: bool):
    """Generate a cost forecast for a building.

    Uses building parameters and optional historical costs to generate
    a probabilistic cost prediction with confidence intervals.

    Example:
        python cli.py predict \\
            --sqft 75000 \\
            --stories 4 \\
            --green-roof \\
            --rooftop-units 8 \\
            --history "150000,162000,148000,175000"
    """
    from app.infrastructure.ml.training_pipeline import TrainingPipeline
    from app.forecasting.models.base_model import BuildingFeatures

    click.echo(click.style('GMP Cost Forecasting - Prediction', fg='cyan', bold=True))

    # Load model
    click.echo(f"Loading model from: {model}")
    pipeline = TrainingPipeline()
    pipeline.load(model)

    # Create features
    features = BuildingFeatures(
        sqft=sqft,
        stories=stories,
        has_green_roof=green_roof,
        rooftop_units_qty=rooftop_units,
        fall_anchor_count=fall_anchors,
    )

    # Parse history
    if history:
        cost_history = np.array([float(x.strip()) for x in history.split(',')])
    else:
        # Use zeros as placeholder if no history provided
        cost_history = np.zeros(pipeline.config.lookback_months)
        click.echo(click.style(
            "Warning: No history provided, using zeros",
            fg='yellow'
        ))

    # Generate prediction
    result = pipeline.predict(
        features=features,
        cost_history=cost_history,
        confidence_level=confidence
    )

    if output_json:
        import json
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(f"\n{'=' * 50}")
        click.echo(click.style("COST FORECAST RESULTS", fg='green', bold=True))
        click.echo(f"{'=' * 50}")
        click.echo(f"Point Estimate:     ${result.point_estimate:>15,.2f}")
        click.echo(f"Lower Bound ({confidence*100:.0f}%): ${result.lower_bound:>15,.2f}")
        click.echo(f"Upper Bound ({confidence*100:.0f}%): ${result.upper_bound:>15,.2f}")
        click.echo(f"Std Deviation:      ${result.std:>15,.2f}")
        click.echo(f"Uncertainty Range:  ${result.uncertainty_range():>15,.2f}")
        click.echo(f"{'=' * 50}")

        # Building info
        click.echo(f"\nBuilding Parameters:")
        click.echo(f"  Square Feet:    {sqft:>12,.0f}")
        click.echo(f"  Stories:        {stories:>12}")
        click.echo(f"  Green Roof:     {'Yes' if green_roof else 'No':>12}")
        click.echo(f"  Rooftop Units:  {rooftop_units:>12}")
        click.echo(f"  Fall Anchors:   {fall_anchors:>12}")
        click.echo(f"  Complexity:     {features.complexity_score:>12.2f}")


@cli.command()
@click.option('--port', type=int, default=8000, help='Server port')
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(port: int, host: str, reload: bool):
    """Start the API server.

    Runs the FastAPI application with uvicorn.

    Example:
        python cli.py serve --port 8000 --reload
    """
    import uvicorn

    click.echo(click.style('GMP Cost Forecasting - API Server', fg='cyan', bold=True))
    click.echo(f"Starting server at http://{host}:{port}")
    click.echo("Press CTRL+C to stop\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload
    )


@cli.command()
@click.option(
    '--model',
    default='models/model.keras',
    help='Path to trained model'
)
def info(model: str):
    """Display information about a trained model.

    Shows model architecture, parameters, and training configuration.

    Example:
        python cli.py info --model models/production.keras
    """
    import json

    click.echo(click.style('GMP Cost Forecasting - Model Info', fg='cyan', bold=True))

    model_path = Path(model)
    if not model_path.exists():
        click.echo(click.style(f"Model not found: {model}", fg='red'))
        click.echo("Train a model first with: python cli.py train ...")
        return

    from app.infrastructure.ml.training_pipeline import TrainingPipeline

    click.echo(f"Loading model from: {model}")
    pipeline = TrainingPipeline()
    pipeline.load(model)

    info = pipeline.get_model_info()

    click.echo(f"\n{'=' * 50}")
    click.echo(click.style("MODEL INFORMATION", fg='green', bold=True))
    click.echo(f"{'=' * 50}")

    if info.get('config'):
        click.echo("\nConfiguration:")
        for key, value in info['config'].items():
            click.echo(f"  {key}: {value}")

    if info.get('model'):
        click.echo("\nModel Details:")
        for key, value in info['model'].items():
            if not isinstance(value, dict):
                click.echo(f"  {key}: {value}")

    if info.get('features', {}).get('fitted'):
        click.echo("\nFeature Statistics:")
        stats = info['features']
        if 'temporal' in stats:
            click.echo(f"  Temporal - mean: {stats['temporal']['mean']:.4f}, std: {stats['temporal']['std']:.4f}")
        if 'target' in stats:
            click.echo(f"  Target - mean: {stats['target']['mean']:.4f}, std: {stats['target']['std']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE-DRIVEN COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

@cli.command('parse-schedule')
@click.option('--data-dir', default='data/raw', help='Directory containing data files')
def parse_schedule(data_dir):
    """Parse schedule and show activity -> trade mappings.

    This is the foundation of schedule-driven forecasting.

    Example:
        python cli.py parse-schedule --data-dir data/raw
    """
    from src.schedule.parser import ScheduleParser
    from src.data.loaders import DataLoader

    click.echo(click.style('Schedule-Driven GMP Forecasting', fg='cyan', bold=True))
    click.echo(f"Loading data from: {data_dir}\n")

    loader = DataLoader(data_dir)
    data = loader.load_all()

    if data['schedule'] is None:
        click.echo(click.style("Error: schedule.csv not found", fg='red'))
        return

    parser = ScheduleParser(data['schedule'])

    click.echo("=" * 90)
    click.echo(click.style("SCHEDULE PARSING RESULTS", fg='green', bold=True))
    click.echo("=" * 90)

    if parser.project_start and parser.project_end:
        click.echo(f"\nProject: {parser.project_start.strftime('%Y-%m-%d')} -> {parser.project_end.strftime('%Y-%m-%d')}")
        click.echo(f"Duration: {(parser.project_end - parser.project_start).days} days")

    click.echo("\n" + "-" * 90)
    click.echo(click.style("PHASES:", bold=True))
    for phase in parser.phases:
        click.echo(f"  {phase.id:20} | {phase.start.strftime('%Y-%m-%d')} -> {phase.end.strftime('%Y-%m-%d')} | {len(phase.activities):3} activities")

    click.echo("\n" + "-" * 90)
    click.echo(click.style("ACTIVITY -> TRADE MAPPINGS (sample):", bold=True))
    click.echo(f"{'Activity ID':<15} {'Activity Name':<40} {'Trade':<30}")
    click.echo("-" * 90)

    for activity in parser.activities[:30]:
        name = activity.name[:38] + '..' if len(activity.name) > 40 else activity.name
        click.echo(f"{activity.id:<15} {name:<40} {activity.primary_trade:<30}")

    if len(parser.activities) > 30:
        click.echo(f"\n... and {len(parser.activities) - 30} more activities")

    # Trade summary
    click.echo("\n" + "-" * 90)
    click.echo(click.style("TRADE SUMMARY:", bold=True))
    summary = parser.get_trade_summary()
    for _, row in summary.sort_values('activity_count', ascending=False).head(15).iterrows():
        click.echo(f"  {row['trade']:<35} {row['activity_count']:3} activities, {row['total_duration_days']:5} total days")


@cli.command('show-allocations')
@click.option('--data-dir', default='data/raw', help='Directory containing data files')
def show_allocations(data_dir):
    """Show expected cost allocations by activity.

    Displays how GMP budget is allocated across schedule activities.

    Example:
        python cli.py show-allocations --data-dir data/raw
    """
    from src.schedule.parser import ScheduleParser
    from src.schedule.cost_allocator import ActivityCostAllocator
    from src.data.loaders import DataLoader

    click.echo(click.style('Schedule-Driven Cost Allocations', fg='cyan', bold=True))

    loader = DataLoader(data_dir)
    data = loader.load_all()

    if data['schedule'] is None or data['breakdown'] is None:
        click.echo(click.style("Error: schedule.csv or breakdown.csv not found", fg='red'))
        return

    parser = ScheduleParser(data['schedule'])
    allocator = ActivityCostAllocator(parser, data['breakdown'])

    click.echo("\n" + "=" * 100)
    click.echo(click.style("ACTIVITY COST ALLOCATIONS (Expected by Schedule)", fg='green', bold=True))
    click.echo("=" * 100)

    for trade, allocs in sorted(
        allocator.allocations.items(),
        key=lambda x: sum(a.expected_cost for a in x[1]),
        reverse=True
    )[:10]:
        total = sum(a.expected_cost for a in allocs)
        click.echo(f"\n{trade}: ${total:,.0f} across {len(allocs)} activities")
        click.echo("-" * 80)

        for alloc in sorted(allocs, key=lambda a: a.expected_cost, reverse=True)[:5]:
            click.echo(f"  {alloc.activity.name[:50]:<52} ${alloc.expected_cost:>12,.0f}")


@cli.command('schedule-train')
@click.option('--data-dir', default='data/raw', help='Directory containing data files')
@click.option('--epochs', default=100, type=int, help='Training epochs')
@click.option('--output', default='models/schedule_driven', help='Output directory')
def schedule_train(data_dir, epochs, output):
    """Train schedule-driven forecasting models.

    Trains models that use schedule position as the primary driver.

    Example:
        python cli.py schedule-train --data-dir data/raw --epochs 100
    """
    from src.training.schedule_driven_trainer import ScheduleDrivenTrainer
    from src.data.loaders import DataLoader

    click.echo(click.style('Schedule-Driven Training Pipeline', fg='cyan', bold=True))

    loader = DataLoader(data_dir)
    data = loader.load_all()

    required = ['schedule', 'breakdown', 'direct_costs']
    missing = [k for k in required if data.get(k) is None]
    if missing:
        click.echo(click.style(f"Error: Missing data files: {missing}", fg='red'))
        return

    trainer = ScheduleDrivenTrainer()
    trainer.prepare(
        schedule_df=data['schedule'],
        gmp_breakdown_df=data['breakdown'],
        direct_costs_df=data['direct_costs'],
        budget_df=data.get('budget')
    )

    click.echo(f"\nTraining models ({epochs} epochs)...")
    results = trainer.train(epochs=epochs)

    # Save models
    trainer.save(output)

    click.echo("\n" + "=" * 80)
    click.echo(click.style("TRAINING COMPLETE", fg='green', bold=True))
    click.echo("=" * 80)
    click.echo(f"\nTrained {len(results)} trade models")
    click.echo(f"Models saved to: {output}")

    for trade, result in results.items():
        click.echo(f"  {trade[:30]:<32} loss={result.final_loss:.4f}, samples={result.samples}")


@cli.command('schedule-forecast')
@click.option('--data-dir', default='data/raw', help='Directory containing data files')
@click.option('--epochs', default=50, type=int, help='Training epochs (quick mode)')
def schedule_forecast(data_dir, epochs):
    """Generate schedule-driven forecasts.

    Trains models and generates forecasts showing schedule variance.

    Example:
        python cli.py schedule-forecast --data-dir data/raw
    """
    from src.training.schedule_driven_trainer import ScheduleDrivenTrainer
    from src.data.loaders import DataLoader

    click.echo(click.style('Schedule-Driven Forecast', fg='cyan', bold=True))

    loader = DataLoader(data_dir)
    data = loader.load_all()

    required = ['schedule', 'breakdown', 'direct_costs']
    missing = [k for k in required if data.get(k) is None]
    if missing:
        click.echo(click.style(f"Error: Missing data files: {missing}", fg='red'))
        return

    trainer = ScheduleDrivenTrainer()
    trainer.prepare(
        schedule_df=data['schedule'],
        gmp_breakdown_df=data['breakdown'],
        direct_costs_df=data['direct_costs']
    )
    trainer.train(epochs=epochs)

    click.echo("\n" + "=" * 110)
    click.echo(click.style("SCHEDULE-DRIVEN FORECAST RESULTS", fg='green', bold=True))
    click.echo("=" * 110)

    click.echo(f"\n{'Trade':<30} {'Phase':<12} {'GMP':>12} {'Spent':>12} {'Expected':>12} {'Variance':>12} {'Status'}")
    click.echo("-" * 110)

    for trade_name in sorted(trainer.models.keys()):
        result = trainer.forecast(trade_name)
        if result:
            variance = result.schedule_variance
            status = "OVER" if variance > 0 else "OK"
            status_color = 'red' if variance > 0 else 'green'
            active = "*" if result.trade_phase_active else " "

            click.echo(
                f"{trade_name[:28]:<30} "
                f"{result.current_phase[:10]:<12} "
                f"${result.gmp_budget:>11,.0f} "
                f"${result.spent_to_date:>11,.0f} "
                f"${result.expected_by_schedule:>11,.0f} "
                f"${variance:>11,.0f} "
                f"{active} {click.style(status, fg=status_color)}"
            )

    click.echo("-" * 110)
    if result:
        click.echo(f"\nProject % Complete: {result.project_pct_complete:.1%}")
    click.echo("\n* = Trade's phase currently active")
    click.echo("Variance = Actual - Expected (positive = over-spending vs schedule)")


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option('--profile', '-p', default=None,
              type=click.Choice(['quick', 'mapping', 'forecasting', 'schedule', 'full']),
              help='Use a predefined calibration profile')
@click.option('--target', '-t', multiple=True, help='Specific target(s) to calibrate')
@click.option('--auto', 'auto_detect', is_flag=True, help='Auto-detect config changes')
@click.option('--dry-run', is_flag=True, help='Show what would be calibrated')
@click.option('--list', 'show_list', is_flag=True, help='List available targets and profiles')
def calibrate(profile, target, auto_detect, dry_run, show_list):
    """Modular calibration for configuration changes.

    Instead of full system retraining, calibrate only the components
    affected by your configuration changes.

    \b
    Examples:
        # Auto-detect config changes and calibrate
        python cli.py calibrate --auto

        # Use quick profile (no model retraining)
        python cli.py calibrate -p quick

        # Calibrate specific targets
        python cli.py calibrate -t fuzzy_matching -t suggestion_engine

        # List available options
        python cli.py calibrate --list

    \b
    Profiles:
        quick       - Parameter updates only (fast, ~10 seconds)
        mapping     - GMP divisions, fuzzy matching, suggestions (~30 seconds)
        forecasting - ML models and parameters (~5 minutes)
        schedule    - Schedule parsing and models (~10 minutes)
        full        - Complete system recalibration (~20 minutes)
    """
    from src.calibration import Calibrator, CalibrationRegistry

    if show_list:
        registry = CalibrationRegistry()
        click.echo(click.style('\nCalibration Targets:', fg='cyan', bold=True))
        click.echo('-' * 60)
        for name, t in sorted(registry.get_all_targets().items()):
            light = '(fast)' if t.is_lightweight else ''
            click.echo(f"  {name:<28} {t.calibration_type:<20} {light}")

        click.echo(click.style('\nCalibration Profiles:', fg='cyan', bold=True))
        click.echo('-' * 60)
        for name, p in sorted(registry.get_all_profiles().items()):
            click.echo(f"  {name:<15} {p.description}")
        return

    calibrator = Calibrator()

    if auto_detect:
        click.echo(click.style('Auto-Detecting Configuration Changes', fg='cyan', bold=True))
        summary = calibrator.auto_calibrate(dry_run=dry_run)
    elif profile:
        click.echo(click.style(f'Calibrating with profile: {profile}', fg='cyan', bold=True))
        summary = calibrator.calibrate_profile(profile, dry_run=dry_run)
    elif target:
        click.echo(click.style(f'Calibrating targets: {", ".join(target)}', fg='cyan', bold=True))
        summary = calibrator.calibrate(list(target), dry_run=dry_run)
    else:
        click.echo(click.style('Auto-Detecting Configuration Changes', fg='cyan', bold=True))
        summary = calibrator.auto_calibrate(dry_run=dry_run)

    if not summary.targets_run:
        click.echo(click.style('\nNo calibration needed. System is up to date.', fg='green'))
        return

    click.echo(f"\nTargets: {len(summary.targets_run)}")
    if summary.triggered_by:
        click.echo(f"Triggered by: {summary.triggered_by}")
    click.echo('-' * 60)

    status_colors = {'completed': 'green', 'failed': 'red', 'skipped': 'yellow'}
    for result in summary.results:
        status = result.status.value
        color = status_colors.get(status, 'white')
        click.echo(f"  {result.target_name:<30} {click.style(status.upper(), fg=color)}")

    click.echo('-' * 60)
    click.echo(f"Completed: {summary.successful_count}/{len(summary.results)} "
               f"in {summary.total_duration_seconds:.1f}s")

    if summary.failed_count > 0:
        click.echo(click.style(f"Failed: {summary.failed_count}", fg='red'))


@cli.command()
def version():
    """Display version information."""
    click.echo(click.style('GMP Cost Forecasting System', fg='cyan', bold=True))
    click.echo("Version: 2.0.0")
    click.echo("")
    click.echo("Architectures:")
    click.echo("  1. Building-Based Forecasting (LSTM/Transformer)")
    click.echo("     - Uses building parameters (sqft, stories, etc.)")
    click.echo("     - Gaussian Mixture / Quantile outputs")
    click.echo("")
    click.echo("  2. Schedule-Driven Forecasting (NEW)")
    click.echo("     - Schedule is PRIMARY driver")
    click.echo("     - Activity -> Trade cost allocation")
    click.echo("     - Schedule variance tracking")
    click.echo("")
    click.echo("Components:")
    click.echo("  - Schedule Parser (activity -> trade mapping)")
    click.echo("  - Activity Cost Allocator (expected cost per activity)")
    click.echo("  - Schedule-Driven Feature Builder")
    click.echo("  - Three-Branch LSTM Model (Schedule + Trade + Cost)")
    click.echo("  - Domain Entities (DirectCost, BudgetLine, GMPAllocation, SubJob)")
    click.echo("  - Cost Mapping Pipeline (Direct -> Budget -> GMP)")
    click.echo("  - LSTM Forecaster (Gaussian Mixture output)")
    click.echo("  - Transformer Forecaster (Quantile outputs)")
    click.echo("")
    click.echo("Dependencies:")
    try:
        import tensorflow as tf
        click.echo(f"  - TensorFlow: {tf.__version__}")
    except ImportError:
        click.echo("  - TensorFlow: Not installed")

    try:
        import numpy as np
        click.echo(f"  - NumPy: {np.__version__}")
    except ImportError:
        click.echo("  - NumPy: Not installed")

    try:
        import pandas as pd
        click.echo(f"  - Pandas: {pd.__version__}")
    except ImportError:
        click.echo("  - Pandas: Not installed")


if __name__ == '__main__':
    cli()
