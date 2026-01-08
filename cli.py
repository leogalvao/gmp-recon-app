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


@cli.command()
def version():
    """Display version information."""
    click.echo(click.style('GMP Cost Forecasting System', fg='cyan', bold=True))
    click.echo("Version: 1.0.0")
    click.echo("")
    click.echo("Components:")
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
