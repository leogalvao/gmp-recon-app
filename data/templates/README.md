# Training Data Templates

## costs_template.csv

Monthly cost records for each project. The model uses 12 months of history to forecast 6 months ahead.

| Column | Type | Description |
|--------|------|-------------|
| `project_id` | string | Unique project identifier |
| `date` | date (YYYY-MM-DD) | First day of the month |
| `cost` | float | Total monthly cost in dollars |

**Requirements:**
- Each project needs at least 12 consecutive months of data
- Dates should be the 1st of each month
- Costs should be positive values

## buildings_template.csv

Building parameters used as static features for the forecasting model.

| Column | Type | Description |
|--------|------|-------------|
| `project_id` | string | Must match project_id in costs.csv |
| `sqft` | float | Total building square footage |
| `stories` | int | Number of floors |
| `has_green_roof` | int (0/1) | Green roof presence (adds 15-25% to roofing costs) |
| `rooftop_units_qty` | int | Number of rooftop HVAC units |
| `fall_anchor_count` | int | Number of fall protection anchors (facade complexity) |

**Derived features (computed automatically):**
- `sqft_per_story`: Average floor plate size
- `complexity_score`: 0-1 scale based on green roof, HVAC units, and fall anchors

## Usage

```bash
# Activate the Python 3.11 venv
source .venv311/bin/activate

# Train the model
python cli.py train \
    --config config/training_config.yaml \
    --data-path data/costs.csv \
    --building-data data/buildings.csv \
    --output models/model.keras

# Generate a prediction
python cli.py predict \
    --model models/model.keras \
    --sqft 75000 \
    --stories 4 \
    --green-roof \
    --rooftop-units 8 \
    --history "150000,162000,148000,175000,168000,155000,182000,195000,178000,165000,172000,188000"
```
