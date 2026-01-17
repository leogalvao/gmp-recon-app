# GMP Reconciliation App

A FastAPI web application for construction project cost reconciliation, mapping direct costs to budget codes and GMP (Guaranteed Maximum Price) divisions with ML-powered forecasting.

## Features

### Core Reconciliation
- **GMP Reconciliation Table** - Main dashboard showing all GMP divisions with actuals, forecasts, and surplus/overrun calculations
- **Regional Allocation** - Split costs between West and East regions with configurable percentages
- **Drill-down Views** - Click any amount to see budget code breakdown with individual records

### Mapping Management
- **Budget to GMP Mapping** - Map budget codes to GMP divisions
- **Direct Cost to Budget Mapping** - Map direct cost entries to budget codes
- **ML-Powered Suggestions** - Intelligent mapping suggestions using fuzzy matching and historical patterns
- **Bulk Accept** - Accept high-confidence suggestions in batch

### Data Quality
- **Duplicate Detection** - Identify potential duplicate entries using exact and fuzzy matching
- **Integrity Validation** - Verify West + East = Total across all divisions
- **Tie-out Checks** - Automated validation that totals reconcile correctly

### Editable Allocations
- **Inline Editing** - Double-click to edit Assigned East/West values
- **Real-time Validation** - Ensures West + East = Total before saving
- **Audit Logging** - Complete change history for all allocation overrides

### ML Forecasting
- **EAC Prediction** - Estimate at Completion using ML models
- **Multiple Forecast Modes** - Actuals only, actuals + commitments, or model-based
- **Configurable EAC Mode** - Choose max, model, or commitments approach

### Multi-Project ML Platform (v1.0+)
- **Hierarchical Transfer Learning** - Train on multiple projects, fine-tune per project
- **Canonical Trade Taxonomy** - 23 CSI division codes for cross-project normalization
- **Probabilistic Forecasting** - Gaussian NLL loss with uncertainty quantification
- **Feature Store** - Normalized cost features (cost/SF, % complete, schedule elapsed)
- **Model Versioning** - DVC integration with S3 remote storage
- **Feature Flags** - Gradual rollout with per-project and percentage-based controls
- **Leakage Prevention** - Temporal and cross-project validation guards

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Jinja2 Templates, jQuery, DataTables
- **ML**: TensorFlow/Keras, scikit-learn
- **Data Processing**: pandas, numpy
- **String Matching**: RapidFuzz
- **Model Versioning**: DVC with S3 remote
- **Auth**: JWT (python-jose, bcrypt)

## Installation

```bash
# Clone the repository
git clone https://github.com/leogalvao/gmp-recon-app.git
cd gmp-recon-app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from app.models import init_db; init_db()"
```

## Usage

### Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

### Access the Application

- **GMP Reconciliation**: http://localhost:8000/gmp
- **Mappings Editor**: http://localhost:8000/mappings
- **Duplicates Review**: http://localhost:8000/duplicates
- **Data Health**: http://localhost:8000/data-health
- **Settings**: http://localhost:8000/settings

### Data Files

Place your data files in the `data/` directory:
- `GMP-Amount.xlsx` - GMP division amounts
- `Budget.xlsx` - Budget codes and committed costs
- `DirectCosts.xlsx` - Direct cost transactions
- `Allocations.xlsx` - Regional allocation percentages

## API Endpoints

### Pages
| Endpoint | Description |
|----------|-------------|
| `GET /gmp` | Main reconciliation table |
| `GET /mappings` | Mapping editor with tabs |
| `GET /duplicates` | Duplicate review interface |
| `GET /data-health` | Data quality dashboard |
| `GET /settings` | Application settings |

### APIs
| Endpoint | Description |
|----------|-------------|
| `GET /api/gmp/drilldown/{division}` | Budget breakdown for a GMP division |
| `GET /api/gmp/relationships` | Data model relationships |
| `GET /api/gmp/allocations/{division}` | Get allocation values |
| `POST /api/gmp/allocations/{division}` | Save allocation override |
| `DELETE /api/gmp/allocations/{division}` | Clear allocation override |
| `GET /api/mappings/suggestions/{id}` | Get mapping suggestions for a direct cost |
| `POST /api/mappings/bulk-accept` | Bulk accept high-confidence mappings |

### ML Pipeline APIs (v1.0+)
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/auth/register` | Register new user |
| `POST /api/v1/auth/login` | Login and get JWT token |
| `GET /api/v1/ml/models` | List registered ML models |
| `GET /api/v1/ml/models/{id}` | Get model details |
| `POST /api/v1/ml/models/{id}/activate` | Set model as production |
| `GET /api/v1/ml/forecast/{project_id}` | Generate project forecast |
| `POST /api/v1/ml/train` | Train global model |
| `GET /api/v1/cutover/status/{project_id}` | Get cutover status |
| `POST /api/v1/cutover/execute/{project_id}` | Execute cutover |
| `GET /api/v1/cutover/feature-flags` | List feature flags |
| `POST /api/v1/cutover/feature-flags/{name}` | Update feature flag |

## Project Structure

```
gmp-recon-app/
├── app/
│   ├── main.py              # FastAPI application and routes
│   ├── models.py            # SQLAlchemy database models
│   ├── api/v1/              # REST API endpoints
│   │   ├── auth.py          # Authentication (JWT)
│   │   ├── ml_pipeline.py   # ML training and inference
│   │   └── cutover.py       # Project cutover management
│   ├── domain/services/     # Business logic services
│   │   ├── model_training_service.py
│   │   ├── forecast_inference_service.py
│   │   ├── training_dataset_service.py
│   │   └── project_cutover_service.py
│   ├── forecasting/models/  # ML model implementations
│   │   └── multi_project_forecaster.py
│   ├── infrastructure/      # Cross-cutting concerns
│   │   └── feature_flags.py
│   ├── modules/
│   │   ├── etl.py           # Data loading and transformation
│   │   ├── mapping.py       # Budget/GMP mapping logic
│   │   ├── reconciliation.py # Core reconciliation calculations
│   │   ├── suggestion_engine.py # ML-powered mapping suggestions
│   │   ├── dedupe.py        # Duplicate detection
│   │   └── ml.py            # Legacy forecasting models
│   └── templates/           # Jinja2 HTML templates
├── models/                  # Trained ML models (DVC tracked)
├── data/                    # Excel data files (gitignored)
├── tests/                   # Test suite (197 tests)
├── alembic/                 # Database migrations
└── requirements.txt         # Python dependencies
```

## Database Models

### Core Models
- **BudgetToGMP** - Mapping from budget codes to GMP divisions
- **DirectToBudget** - Mapping from direct costs to budget codes
- **Allocation** - Regional allocation splits (West/East percentages)
- **GMPAllocationOverride** - Manual overrides for computed allocations
- **AllocationChangeLog** - Audit trail for allocation changes
- **Duplicate** - Detected duplicate entries
- **Settings** - Application configuration
- **Run** - Reconciliation run history

### Multi-Project Models (v1.0+)
- **Project** - Project metadata with training eligibility
- **GMP** - GMP divisions with canonical trade mapping
- **CanonicalTrade** - 23 CSI division taxonomy
- **CanonicalCostFeature** - Normalized cost features per project/trade/period
- **MLModelRegistry** - Trained model versions and metadata
- **ProjectForecast** - Per-trade forecast predictions
- **FeatureFlagState** - Feature flag persistence
- **User** - Authentication users

## ML Model Architecture

The Multi-Project Forecaster uses hierarchical transfer learning:

```
Input: [sequence_features, project_id, trade_id]
         ↓
┌─────────────────────────────────────┐
│  Project Embedding (32-dim)         │
│  Trade Embedding (16-dim)           │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Bidirectional LSTM (64 units)      │
│  + Dropout (0.2)                    │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Project-Specific Adapter (32 units)│
│  Gated residual connection          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Output: [mean, log_variance]       │
│  Gaussian NLL Loss                  │
└─────────────────────────────────────┘
```

**Parameters**: ~150K | **Input Features**: 5 | **Sequence Length**: 12 months

## Running Tests

```bash
pytest tests/ -v
```

## Model Versioning with DVC

Models are tracked with DVC and stored in S3:

```bash
# Pull trained models
dvc pull

# After training a new model
dvc add models/my_model
dvc push

# Check model status
dvc status
```

## CLI Usage

```bash
# Activate virtual environment
source .venv311/bin/activate

# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Train ML model (via API)
curl -X POST http://localhost:8000/api/v1/ml/train \
  -H "Authorization: Bearer $TOKEN"

# Generate forecast
curl http://localhost:8000/api/v1/ml/forecast/2 \
  -H "Authorization: Bearer $TOKEN"

# Check feature flags
curl http://localhost:8000/api/v1/cutover/feature-flags \
  -H "Authorization: Bearer $TOKEN"
```

## License

MIT License
