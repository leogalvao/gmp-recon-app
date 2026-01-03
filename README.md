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

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Jinja2 Templates, jQuery, DataTables
- **ML**: scikit-learn, PyTorch (optional)
- **Data Processing**: pandas, numpy
- **String Matching**: RapidFuzz

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

## Project Structure

```
gmp-recon-app/
├── app/
│   ├── main.py              # FastAPI application and routes
│   ├── models.py            # SQLAlchemy database models
│   ├── modules/
│   │   ├── etl.py           # Data loading and transformation
│   │   ├── mapping.py       # Budget/GMP mapping logic
│   │   ├── reconciliation.py # Core reconciliation calculations
│   │   ├── suggestion_engine.py # ML-powered mapping suggestions
│   │   ├── dedupe.py        # Duplicate detection
│   │   └── ml.py            # Forecasting models
│   └── templates/           # Jinja2 HTML templates
├── data/                    # Excel data files (gitignored)
├── tests/                   # Test suite
├── alembic/                 # Database migrations
└── requirements.txt         # Python dependencies
```

## Database Models

- **BudgetToGMP** - Mapping from budget codes to GMP divisions
- **DirectToBudget** - Mapping from direct costs to budget codes
- **Allocation** - Regional allocation splits (West/East percentages)
- **GMPAllocationOverride** - Manual overrides for computed allocations
- **AllocationChangeLog** - Audit trail for allocation changes
- **Duplicate** - Detected duplicate entries
- **Settings** - Application configuration
- **Run** - Reconciliation run history

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License
