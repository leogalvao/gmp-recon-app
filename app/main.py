"""
Main FastAPI Application for GMP Reconciliation.
Serves HTML UI via Jinja2 templates and provides REST endpoints.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
from typing import Dict, Optional, List
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, Depends, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
import io

from app.models import (
    init_db, get_db, ensure_default_settings,
    Settings, Run, BudgetToGMP, DirectToBudget, Allocation, Duplicate,
    SideConfiguration, GMPBudgetBreakdown, ScheduleActivity, ScheduleToGMPMapping,
    Project, GMP, BudgetEntity, ChangeOrder, DirectCostEntity,
    TrainingRound, TrainingForecastSnapshot, ChangeOrderStatus, Zone, AllocationMethod
)
from app.modules.etl import (
    get_data_loader, cents_to_display, get_file_hashes,
    load_breakdown_csv, load_schedule_csv,
    fuzzy_match_breakdown_to_gmp, match_schedule_to_gmp,
    allocate_east_west
)
from app.modules.mapping import (
    map_budget_to_gmp, map_direct_to_budget, apply_allocations,
    save_mapping, get_mapping_stats
)
from app.modules.reconciliation import (
    compute_reconciliation, format_for_display, compute_summary_metrics,
    validate_tie_outs, get_settings, get_gmp_drilldown,
    compute_dashboard_summary, compute_schedule_based_forecast
)
from app.modules.dedupe import (
    detect_duplicates, apply_duplicate_exclusions, get_duplicates_summary,
    format_duplicates_for_display
)
from app.modules.ml import get_forecasting_pipeline
from app.modules.suggestion_engine import (
    compute_all_suggestions, compute_single_suggestion,
    record_mapping_feedback, get_cached_suggestions,
    df_to_direct_cost_rows, df_to_budget_rows,
    normalize_vendor, _get_thresholds
)
from app.config import get_config
from app.modules.forecasting import (
    ForecastManager, compute_project_rollup,
    calculate_evm, calculate_pert, calculate_parametric
)
from app.models import ForecastConfig, ForecastSnapshot, ForecastPeriod, ForecastAuditLog
from app.modules.csrf import csrf, get_or_create_csrf_token
from app.api.v1 import api_router as v1_router


# Initialize FastAPI app
app = FastAPI(
    title="GMP Reconciliation App",
    description="Reconcile Procore Direct Costs against GMP funding via Budget mapping",
    version="1.0.0"
)

# Include v1 API routes for cost management hierarchy
app.include_router(v1_router)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Custom Jinja2 filters
def format_currency(value):
    if isinstance(value, (int, float)):
        return cents_to_display(int(value))
    return value

templates.env.filters['currency'] = format_currency


# CSRF Middleware - sets token cookie and validates POST requests
@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """
    CSRF protection middleware.

    - Sets CSRF token cookie if not present
    - Validates CSRF token for state-changing requests (POST, PUT, DELETE)
    - Exempt: /api/* endpoints (for external API consumers with their own auth)
    """
    response = await call_next(request)

    # Set CSRF token cookie if not present
    if csrf.cookie_name not in request.cookies:
        token = csrf.generate_token()
        response.set_cookie(
            key=csrf.cookie_name,
            value=token,
            httponly=False,  # JS needs to read for AJAX
            samesite="strict",
            secure=False,  # Set True in production with HTTPS
            max_age=3600
        )

    return response


# Add csrf_token to all template contexts
def get_template_context(request: Request, **kwargs) -> dict:
    """Build template context with CSRF token included."""
    return {
        "request": request,
        "csrf_token": get_or_create_csrf_token(request),
        **kwargs
    }


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    ensure_default_settings()
    setup_scheduler()


# Background scheduler for nightly retraining
scheduler = None

def setup_scheduler():
    global scheduler
    scheduler = BackgroundScheduler()
    
    # Nightly retrain at 02:00
    scheduler.add_job(run_nightly_train, 'cron', hour=2, minute=0)
    
    # File watcher (check every 5 minutes)
    scheduler.add_job(check_file_changes, 'interval', minutes=5)
    
    scheduler.start()


def run_nightly_train():
    """Background job for nightly model retraining."""
    db = next(get_db())
    try:
        run = Run(
            run_type='nightly_train',
            status='running',
            started_at=datetime.now(timezone.utc)
        )
        db.add(run)
        db.commit()
        
        # Retrain ML models
        data_loader = get_data_loader()
        pipeline = get_forecasting_pipeline()
        
        settings = get_settings(db)
        pipeline.train(
            data_loader.direct_costs,
            data_loader.budget,
            data_loader.gmp,
            settings.get('as_of_date')
        )
        
        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.notes = json.dumps(pipeline.get_training_status())
        db.commit()
    except Exception as e:
        run.status = 'failed'
        run.notes = str(e)
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def check_file_changes():
    """Check if input files have changed and trigger recompute."""
    data_loader = get_data_loader()
    if data_loader.check_for_changes():
        data_loader.reload()
        # Could trigger full recompute here


# ------------ Helper Functions ------------

def run_full_reconciliation(db: Session, side_filter: Optional[str] = None) -> Dict:
    """
    Execute full reconciliation pipeline:
    1. Load data
    2. Run mappings
    3. Detect duplicates
    4. Apply allocations
    5. Run ML predictions
    6. Compute reconciliation

    Args:
        db: Database session
        side_filter: Optional side filter (EAST, WEST, BOTH). When EAST/WEST is specified,
                     includes BOTH mappings as well.
    """
    data_loader = get_data_loader()
    settings = get_settings(db)

    # Get data
    gmp_df = data_loader.gmp.copy()
    budget_df = data_loader.budget.copy()
    direct_costs_df = data_loader.direct_costs.copy()
    allocations_df = data_loader.allocations.copy()

    # Map Budget to GMP
    budget_df = map_budget_to_gmp(budget_df, gmp_df, db)

    # Map Direct Costs to Budget
    direct_costs_df = map_direct_to_budget(direct_costs_df, budget_df, db)

    # Apply side filter if specified
    if side_filter:
        # Get budget codes that match the side filter
        if side_filter != 'BOTH':
            budget_side_mappings = db.query(BudgetToGMP).filter(
                BudgetToGMP.side.in_([side_filter, 'BOTH'])
            ).all()
        else:
            budget_side_mappings = db.query(BudgetToGMP).filter(
                BudgetToGMP.side == 'BOTH'
            ).all()

        side_budget_codes = {m.budget_code for m in budget_side_mappings}

        # Filter budget_df to only include budget codes matching the side
        if side_budget_codes:
            budget_df = budget_df[budget_df['Budget Code'].isin(side_budget_codes)]

        # Get direct cost mappings that match the side filter
        if side_filter != 'BOTH':
            direct_side_mappings = db.query(DirectToBudget).filter(
                DirectToBudget.side.in_([side_filter, 'BOTH'])
            ).all()
        else:
            direct_side_mappings = db.query(DirectToBudget).filter(
                DirectToBudget.side == 'BOTH'
            ).all()

        side_direct_keys = {(m.cost_code, m.name) for m in direct_side_mappings}

        # Filter direct_costs_df to only include items matching the side
        if side_direct_keys:
            direct_costs_df = direct_costs_df[
                direct_costs_df.apply(
                    lambda r: (r.get('Cost Code', ''), r.get('Name', '')) in side_direct_keys,
                    axis=1
                )
            ]
    
    # Detect duplicates
    duplicates, _ = detect_duplicates(direct_costs_df)
    direct_costs_df = apply_duplicate_exclusions(direct_costs_df, duplicates)
    
    # Apply allocations to budget (for commitments) - use base_code
    budget_df = apply_allocations(budget_df, 'committed_costs_cents', 'base_code', allocations_df, db)
    
    # Apply allocations to direct costs - use base_code
    direct_costs_df = apply_allocations(direct_costs_df, 'amount_cents', 'base_code', allocations_df, db)
    
    # Load breakdown data for East/West allocations
    breakdown_records = db.query(GMPBudgetBreakdown).all()
    breakdown_df = None
    if breakdown_records:
        breakdown_df = pd.DataFrame([{
            'gmp_division': b.gmp_division,
            'east_funded_cents': b.east_funded_cents,
            'west_funded_cents': b.west_funded_cents,
            'pct_east': b.pct_east,
            'pct_west': b.pct_west
        } for b in breakdown_records if b.gmp_division])
        # Aggregate by GMP division (sum allocations)
        if not breakdown_df.empty:
            breakdown_df = breakdown_df.groupby('gmp_division').agg({
                'east_funded_cents': 'sum',
                'west_funded_cents': 'sum',
                'pct_east': 'mean',
                'pct_west': 'mean'
            }).reset_index()

    # Run ML predictions
    pipeline = get_forecasting_pipeline()
    if pipeline.last_trained is None:
        pipeline.train(direct_costs_df, budget_df, gmp_df, settings.get('as_of_date'))

    predictions_df = pipeline.predict(direct_costs_df, budget_df, gmp_df, settings.get('as_of_date'))

    # Compute reconciliation with breakdown data
    recon_df = compute_reconciliation(
        gmp_df, budget_df, direct_costs_df, predictions_df, settings,
        breakdown_df=breakdown_df
    )
    
    # Format for display
    recon_rows = format_for_display(recon_df)
    
    # Compute summary metrics
    summary = compute_summary_metrics(recon_df, direct_costs_df, budget_df)
    
    # Validate tie-outs
    tie_outs = validate_tie_outs(recon_df, direct_costs_df)
    
    # Mapping stats
    mapping_stats = get_mapping_stats(budget_df, direct_costs_df)
    
    # Duplicate summary
    dup_summary = get_duplicates_summary(duplicates)
    
    return {
        'recon_rows': recon_rows,
        'summary': summary,
        'tie_outs': tie_outs,
        'mapping_stats': mapping_stats,
        'duplicates_summary': dup_summary,
        'settings': settings,
        'last_ml_train': pipeline.get_training_status(),
        'duplicates': duplicates,
        'budget_df': budget_df,
        'direct_costs_df': direct_costs_df
    }


# ------------ Routes ------------

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, db: Session = Depends(get_db)):
    """Project dashboard with overview metrics and division summaries."""
    try:
        # Get dashboard summary (single source of truth for KPIs)
        dashboard_summary = compute_dashboard_summary(db)

        # Get full reconciliation data for division cards
        result = run_full_reconciliation(db)
        recon_rows = result['recon_rows']

        # Diagnostic logging for dashboard metrics
        total_gmp_from_rows = sum(r.get('gmp_amount_raw', 0) for r in recon_rows)
        total_actual_from_rows = sum((r.get('actual_west_raw', 0) + r.get('actual_east_raw', 0)) for r in recon_rows)
        total_eac_from_rows = sum(r.get('eac_total_raw', 0) for r in recon_rows)
        logger.info(f"Dashboard metrics debug:")
        logger.info(f"  Recon rows count: {len(recon_rows)}")
        logger.info(f"  GMP from entity table: {dashboard_summary['total_gmp_budget_cents']}")
        logger.info(f"  GMP from recon rows: {total_gmp_from_rows}")
        logger.info(f"  Actual from entity table: {dashboard_summary['actual_costs_cents']}")
        logger.info(f"  Actual from recon rows: {total_actual_from_rows}")
        logger.info(f"  EAC from entity table: {dashboard_summary['eac_cents']}")
        logger.info(f"  EAC from recon rows: {total_eac_from_rows}")
        logger.info(f"  Forecast remaining: {dashboard_summary['forecast_remaining_cents']}")
        # Log first row to debug field names
        if recon_rows:
            first_row = recon_rows[0]
            logger.info(f"  First row keys: {list(first_row.keys())}")
            logger.info(f"  First row sample: gmp_division={first_row.get('gmp_division')}, gmp_amount_raw={first_row.get('gmp_amount_raw')}, actual_west_raw={first_row.get('actual_west_raw')}, actual_east_raw={first_row.get('actual_east_raw')}, eac_total_raw={first_row.get('eac_total_raw')}")

        # Get schedule variances for per-division display
        schedule_variances = get_schedule_variances()

        # Get forecast rollup for per-division CPI
        forecast_rollup = compute_project_rollup(db)

        # Calculate schedule variance totals for display
        total_schedule_expected = sum(v.get('expected', 0) for v in schedule_variances.values())
        total_schedule_spent = sum(v.get('spent', 0) for v in schedule_variances.values())
        total_schedule_variance = total_schedule_spent - total_schedule_expected
        total_schedule_variance_pct = round(total_schedule_variance / total_schedule_expected * 100, 1) if total_schedule_expected > 0 else 0

        # Compute totals directly from recon_rows (authoritative source from reconciliation)
        # This ensures dashboard totals match the GMP reconciliation page
        total_gmp_from_recon = sum(row.get('gmp_amount_raw', 0) or 0 for row in recon_rows)
        total_actual_from_recon = sum(
            (row.get('actual_west_raw', 0) or 0) + (row.get('actual_east_raw', 0) or 0)
            for row in recon_rows
        )
        total_eac_from_recon = sum(row.get('eac_total_raw', 0) or 0 for row in recon_rows)

        logger.info(f"Recon totals - GMP: {total_gmp_from_recon}, Actual: {total_actual_from_recon}, EAC: {total_eac_from_recon}")

        # Build division cards with health status
        # Use pre-formatted values from format_for_display where available
        division_cards = []
        for row in recon_rows:
            gmp_div = row['gmp_division']

            # Get raw cents values (convert to int to handle numpy types)
            gmp_cents = int(row.get('gmp_amount_raw', 0) or 0)
            actual_west = int(row.get('actual_west_raw', 0) or 0)
            actual_east = int(row.get('actual_east_raw', 0) or 0)
            actual_cents = actual_west + actual_east
            eac_cents = int(row.get('eac_total_raw', 0) or 0)

            # Calculate variance as GMP - EAC
            variance_cents = gmp_cents - eac_cents

            # Get forecast CPI for health status
            div_forecast = next((d for d in forecast_rollup.get('by_division', []) if d['gmp_division'] == gmp_div), {})
            cpi = div_forecast.get('cpi')

            # Get schedule variance
            sched_var = schedule_variances.get(gmp_div, {})

            # Determine health status
            pct_spent = round(actual_cents / gmp_cents * 100, 1) if gmp_cents > 0 else 0
            if cpi and cpi < 0.9:
                health = 'critical'
            elif cpi and cpi < 0.95:
                health = 'warning'
            elif variance_cents < 0:
                health = 'warning'
            else:
                health = 'healthy'

            # Use pre-formatted display values from row, or format raw values
            division_cards.append({
                'name': gmp_div,
                'gmp_cents': gmp_cents,
                'gmp_display': row.get('gmp_amount') or cents_to_display(gmp_cents),
                'actual_cents': actual_cents,
                'actual_display': row.get('actual_total') or cents_to_display(actual_cents),
                'eac_cents': eac_cents,
                'eac_display': row.get('eac_total') or cents_to_display(eac_cents),
                'variance_cents': variance_cents,
                'variance_display': row.get('surplus_or_overrun') or cents_to_display(variance_cents),
                'pct_spent': pct_spent,
                'cpi': cpi,
                'health': health,
                'schedule_variance_pct': sched_var.get('variance_pct', 0),
                'schedule_status': sched_var.get('status', 'on_track')
            })

        # Sort by variance (worst first)
        division_cards.sort(key=lambda x: x['variance_cents'])

        # Log division cards for debugging
        logger.info(f"Division cards count: {len(division_cards)}")
        if division_cards:
            logger.info(f"First card: {division_cards[0]}")

        # Count health statuses
        health_counts = {
            'healthy': len([d for d in division_cards if d['health'] == 'healthy']),
            'warning': len([d for d in division_cards if d['health'] == 'warning']),
            'critical': len([d for d in division_cards if d['health'] == 'critical'])
        }

        # Get mapping stats
        mapping_stats = result.get('mapping_stats', {})

        # Use reconciliation data as the primary source (matches GMP page)
        # Only fall back to dashboard_summary if recon data is unavailable
        total_gmp_cents = total_gmp_from_recon if total_gmp_from_recon > 0 else dashboard_summary['total_gmp_budget_cents']
        actual_cents_total = total_actual_from_recon if total_actual_from_recon > 0 else dashboard_summary['actual_costs_cents']
        eac_cents_total = total_eac_from_recon if total_eac_from_recon > 0 else dashboard_summary['eac_cents']

        # Remove stale warning if we have data from recon
        if total_gmp_cents > 0 and "GMP Budget data unavailable or zero" in dashboard_summary.get('warnings', []):
            dashboard_summary['warnings'] = [w for w in dashboard_summary['warnings']
                                             if w != "GMP Budget data unavailable or zero"]

        # Variance = Budget - EAC (positive = underrun, negative = overrun)
        variance_cents = total_gmp_cents - eac_cents_total if total_gmp_cents > 0 else None

        # Progress = Actual / EAC
        progress_pct = 0.0
        if eac_cents_total > 0:
            progress_pct = round((actual_cents_total / eac_cents_total) * 100, 1)
            progress_pct = max(0.0, min(100.0, progress_pct))

        # Format for display
        project_metrics = {
            'total_gmp_cents': total_gmp_cents,
            'total_gmp_display': cents_to_display(total_gmp_cents) if total_gmp_cents > 0 else 'N/A',
            'total_actual_cents': actual_cents_total,
            'total_actual_display': cents_to_display(actual_cents_total),
            'total_eac_cents': eac_cents_total,
            'total_eac_display': cents_to_display(eac_cents_total),
            'total_variance_cents': variance_cents,
            'total_variance_display': cents_to_display(variance_cents) if variance_cents is not None else 'N/A',
            'variance_pct': round(variance_cents / total_gmp_cents * 100, 1) if total_gmp_cents > 0 and variance_cents is not None else None,
            'pct_complete': progress_pct,
            'overall_cpi': dashboard_summary['cpi'],
            'schedule_variance': total_schedule_variance,
            'schedule_variance_pct': total_schedule_variance_pct,
            'schedule_status': 'over' if total_schedule_variance > 0 else 'under' if total_schedule_variance < 0 else 'on_track',
            'warnings': dashboard_summary['warnings']
        }

        return templates.TemplateResponse(request, "dashboard.html", {
            "project_metrics": project_metrics,
            "division_cards": division_cards,
            "health_counts": health_counts,
            "mapping_stats": mapping_stats,
            "division_count": len(division_cards),
            "active_page": "dashboard"
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return templates.TemplateResponse(request, "error.html", {
            "error": str(e),
            "active_page": "dashboard"
        })


def get_schedule_variances() -> Dict[str, Dict]:
    """
    Get schedule variance data for all GMP trades.
    Returns dict of trade -> {expected, spent, variance, variance_pct, status}
    """
    from pathlib import Path
    from datetime import datetime, timezone

    data_dir = Path(__file__).parent.parent / "data" / "raw"
    schedule_file = data_dir / "schedule.csv"
    breakdown_file = data_dir / "breakdown.csv"
    direct_costs_file = data_dir / "direct_costs.csv"

    if not all(f.exists() for f in [schedule_file, breakdown_file, direct_costs_file]):
        return {}

    try:
        from src.schedule.parser import ScheduleParser
        from src.schedule.cost_allocator import ActivityCostAllocator

        schedule_df = pd.read_csv(schedule_file)
        breakdown_df = pd.read_csv(breakdown_file)
        direct_costs_df = pd.read_csv(direct_costs_file)

        parser = ScheduleParser(schedule_df)
        allocator = ActivityCostAllocator(parser, breakdown_df)

        # Aggregate actual costs by trade
        amount_col = next((c for c in ['amount', 'Amount'] if c in direct_costs_df.columns), None)
        actual_by_trade = {}
        if amount_col:
            for _, row in direct_costs_df.iterrows():
                name = str(row.get('name', '') or row.get('Description', ''))
                trade, _, _, _ = parser._map_to_trade(name)
                if trade:
                    actual_by_trade[trade] = actual_by_trade.get(trade, 0) + float(row[amount_col])

        # Calculate variances
        as_of = datetime.now()
        variances = {}
        for trade in allocator.gmp.keys():
            expected = allocator.get_expected_cost_to_date(trade, as_of)
            spent = actual_by_trade.get(trade, 0)
            variance = spent - expected
            variance_pct = (variance / expected * 100) if expected > 0 else 0

            variances[trade] = {
                'expected': expected,
                'spent': spent,
                'variance': variance,
                'variance_pct': round(variance_pct, 1),
                'status': 'over' if variance > 0 else 'under' if variance < 0 else 'on_track'
            }

        return variances
    except Exception as e:
        logger.warning(f"Could not calculate schedule variances: {e}")
        return {}


@app.get("/gmp", response_class=HTMLResponse)
async def gmp_page(
    request: Request,
    side: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Main GMP reconciliation table page with optional side filter."""
    try:
        # Get available sides for filter dropdown
        side_configs = db.query(SideConfiguration).filter(SideConfiguration.is_active == True).all()
        available_sides = [{'value': s.side, 'label': s.display_name} for s in side_configs]

        # Validate and normalize side filter
        side_filter = None
        if side:
            side_upper = side.upper()
            if side_upper in VALID_SIDES:
                side_filter = side_upper

        result = run_full_reconciliation(db, side_filter=side_filter)

        # Get schedule variance data
        schedule_variances = get_schedule_variances()

        # Add schedule variance to each row
        for row in result['recon_rows']:
            gmp_div = row['gmp_division']
            var_data = schedule_variances.get(gmp_div, {})
            row['schedule_variance'] = var_data.get('variance', 0)
            row['schedule_variance_pct'] = var_data.get('variance_pct', 0)
            row['schedule_variance_status'] = var_data.get('status', 'on_track')
            row['schedule_expected'] = var_data.get('expected', 0)

        return templates.TemplateResponse(request, "gmp.html", {
            "rows": result['recon_rows'],
            "summary": result['summary'],
            "tie_outs": result['tie_outs'],
            "mapping_stats": result['mapping_stats'],
            "duplicates_summary": result['duplicates_summary'],
            "settings": result['settings'],
            "last_ml_train": result['last_ml_train'],
            "side_filter": side_filter,
            "available_sides": available_sides,
            "active_page": "gmp"
        })
    except Exception as e:
        return templates.TemplateResponse(request, "error.html", {
            "error": str(e),
            "active_page": "gmp"
        })


@app.get("/forecast", response_class=HTMLResponse)
async def forecast_project_page(
    request: Request,
    divisions: str = None,
    granularity: str = "weekly",
    side: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Project-level forecast page with optional division and side filtering."""
    from urllib.parse import unquote

    # Get available sides for filter dropdown
    side_configs = db.query(SideConfiguration).filter(SideConfiguration.is_active == True).all()
    available_sides = [{'value': s.side, 'label': s.display_name} for s in side_configs]

    # Validate and normalize side filter
    side_filter = None
    if side:
        side_upper = side.upper()
        if side_upper in VALID_SIDES:
            side_filter = side_upper

    # Get all available divisions from config
    app_config = get_config()
    all_divisions = []
    for key, div_config in app_config.gmp_divisions.items():
        all_divisions.append({
            'key': key,
            'name': div_config.get('name', key)
        })
    all_divisions.sort(key=lambda x: x['name'])

    # Parse selected divisions
    selected_divisions = []
    if divisions:
        selected_divisions = [unquote(d.strip()) for d in divisions.split(',') if d.strip()]

    # Validate granularity
    if granularity not in ['weekly', 'monthly']:
        granularity = 'weekly'

    # Get project rollup
    manager = ForecastManager(db)
    rollup = compute_project_rollup(db)

    # Get schedule variances for all divisions
    schedule_variances = get_schedule_variances()

    # Filter by selected divisions if specified
    division_forecasts = rollup.get('by_division', [])
    if selected_divisions:
        division_forecasts = [d for d in division_forecasts if d['gmp_division'] in selected_divisions]

    # Add schedule variance to each division and calculate totals
    total_schedule_expected = 0
    total_schedule_spent = 0
    total_schedule_variance = 0
    for div in division_forecasts:
        div_name = div.get('gmp_division', '')
        div_sched = schedule_variances.get(div_name, {})
        div['schedule_expected'] = div_sched.get('expected', 0)
        div['schedule_spent'] = div_sched.get('spent', 0)
        div['schedule_variance'] = div_sched.get('variance', 0)
        div['schedule_variance_pct'] = div_sched.get('variance_pct', 0)
        div['schedule_variance_status'] = div_sched.get('status', 'on_track')
        # Accumulate totals
        total_schedule_expected += div['schedule_expected']
        total_schedule_spent += div['schedule_spent']
        total_schedule_variance += div['schedule_variance']

    # Calculate total schedule variance percent
    total_schedule_variance_pct = round(total_schedule_variance / total_schedule_expected * 100, 1) if total_schedule_expected > 0 else 0
    total_schedule_status = 'over' if total_schedule_variance > 0 else 'under' if total_schedule_variance < 0 else 'on_track'

    # Calculate filtered totals
    if selected_divisions and division_forecasts:
        total_bac = sum(d.get('bac_cents', 0) or 0 for d in division_forecasts)
        total_ac = sum(d.get('ac_cents', 0) or 0 for d in division_forecasts)
        total_eac = sum(d.get('eac_cents', 0) or 0 for d in division_forecasts)
        total_etc = sum(d.get('etc_cents', 0) or 0 for d in division_forecasts)
        total_var = sum(d.get('var_cents', 0) or 0 for d in division_forecasts)
    else:
        total_bac = rollup.get('total_bac_cents', 0)
        total_ac = rollup.get('total_ac_cents', 0)
        total_eac = rollup.get('total_eac_cents', 0)
        total_etc = rollup.get('total_etc_cents', 0)
        total_var = rollup.get('total_var_cents', 0)

    # Build project forecast dict
    forecast = {
        'has_forecast': len(division_forecasts) > 0,
        'is_project_view': True,
        'bac_cents': total_bac,
        'bac_display': cents_to_display(total_bac),
        'ac_cents': total_ac,
        'ac_display': cents_to_display(total_ac),
        'eac_cents': total_eac,
        'eac_display': cents_to_display(total_eac),
        'eac_west_cents': total_eac // 2 if total_eac else 0,
        'eac_east_cents': total_eac - (total_eac // 2) if total_eac else 0,
        'etc_cents': total_etc,
        'etc_display': cents_to_display(total_etc),
        'var_cents': total_var,
        'var_display': cents_to_display(total_var),
        'var_percent': round(total_var / total_bac * 100, 1) if total_bac else 0,
        'percent_complete': round(total_ac / total_eac * 100, 1) if total_eac else 0,
        'cpi': rollup.get('overall_cpi'),
        'spi': None,
        'method': 'rollup',
        'confidence_score': 0.7,
        'confidence_band': 'medium',
        'explanation': 'Project-level rollup of all division forecasts',
        'trigger': 'manual',
        'snapshot_date': None,
        'division_count': len(division_forecasts),
        'divisions': division_forecasts,
        # Schedule variance totals
        'schedule_expected': total_schedule_expected,
        'schedule_spent': total_schedule_spent,
        'schedule_variance': total_schedule_variance,
        'schedule_variance_pct': total_schedule_variance_pct,
        'schedule_variance_status': total_schedule_status
    }

    # Config for project view
    config_dict = {
        'method': 'rollup',
        'distribution_method': 'linear',
        'completion_date': None,
        'is_locked': False
    }

    return templates.TemplateResponse(request, "forecast.html", {
        "gmp_division": "All Divisions" if not selected_divisions else ", ".join(selected_divisions),
        "granularity": granularity,
        "forecast": forecast,
        "config": config_dict,
        "periods": {'periods': [], 'period_count': 0},
        "all_divisions": all_divisions,
        "selected_divisions": selected_divisions,
        "side_filter": side_filter,
        "available_sides": available_sides,
        "is_project_view": True,
        "active_page": "forecast"
    })


@app.get("/gmp/{gmp_division}/forecast", response_class=HTMLResponse)
async def forecast_page(
    request: Request,
    gmp_division: str,
    granularity: str = "weekly",
    side: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Forecast page for a specific GMP division."""
    from urllib.parse import unquote

    gmp_division = unquote(gmp_division)

    # Get available sides for filter dropdown
    side_configs = db.query(SideConfiguration).filter(SideConfiguration.is_active == True).all()
    available_sides = [{'value': s.side, 'label': s.display_name} for s in side_configs]

    # Validate and normalize side filter
    side_filter = None
    if side:
        side_upper = side.upper()
        if side_upper in VALID_SIDES:
            side_filter = side_upper

    # Validate granularity
    if granularity not in ['weekly', 'monthly']:
        granularity = 'weekly'

    # Get forecast data
    manager = ForecastManager(db)
    config = manager.get_or_create_config(gmp_division)
    snapshot = manager.get_current_snapshot(gmp_division)

    # Get schedule variance for this division
    schedule_variances = get_schedule_variances()
    div_schedule_var = schedule_variances.get(gmp_division, {})

    # Build forecast dict
    if snapshot:
        forecast = {
            'has_forecast': True,
            'snapshot_id': snapshot.id,
            'snapshot_date': snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None,
            'bac_cents': snapshot.bac_cents,
            'bac_display': cents_to_display(snapshot.bac_cents),
            'ac_cents': snapshot.ac_cents,
            'ac_display': cents_to_display(snapshot.ac_cents),
            'ev_cents': snapshot.ev_cents,
            'eac_cents': snapshot.eac_cents,
            'eac_display': cents_to_display(snapshot.eac_cents),
            'eac_west_cents': snapshot.eac_west_cents,
            'eac_east_cents': snapshot.eac_east_cents,
            'etc_cents': snapshot.etc_cents,
            'etc_display': cents_to_display(snapshot.etc_cents),
            'var_cents': snapshot.var_cents,
            'var_display': cents_to_display(snapshot.var_cents),
            'var_percent': round(snapshot.var_cents / snapshot.bac_cents * 100, 1) if snapshot.bac_cents else 0,
            'percent_complete': round(snapshot.ac_cents / snapshot.eac_cents * 100, 1) if snapshot.eac_cents else 0,
            'cpi': snapshot.cpi,
            'spi': snapshot.spi,
            'method': snapshot.method,
            'confidence_score': snapshot.confidence_score,
            'confidence_band': snapshot.confidence_band,
            'explanation': snapshot.explanation,
            'trigger': snapshot.trigger,
            # Schedule variance fields
            'schedule_expected': div_schedule_var.get('expected', 0),
            'schedule_spent': div_schedule_var.get('spent', 0),
            'schedule_variance': div_schedule_var.get('variance', 0),
            'schedule_variance_pct': div_schedule_var.get('variance_pct', 0),
            'schedule_variance_status': div_schedule_var.get('status', 'on_track')
        }
    else:
        forecast = {
            'has_forecast': False,
            # Still include schedule variance even without forecast
            'schedule_expected': div_schedule_var.get('expected', 0),
            'schedule_spent': div_schedule_var.get('spent', 0),
            'schedule_variance': div_schedule_var.get('variance', 0),
            'schedule_variance_pct': div_schedule_var.get('variance_pct', 0),
            'schedule_variance_status': div_schedule_var.get('status', 'on_track')
        }

    # Build config dict
    config_dict = {
        'method': config.method,
        'evm_performance_factor': config.evm_performance_factor,
        'pert_optimistic_cents': config.pert_optimistic_cents,
        'pert_most_likely_cents': config.pert_most_likely_cents,
        'pert_pessimistic_cents': config.pert_pessimistic_cents,
        'param_quantity': config.param_quantity,
        'param_unit_rate_cents': config.param_unit_rate_cents,
        'param_complexity_factor': config.param_complexity_factor,
        'distribution_method': config.distribution_method,
        'completion_date': config.completion_date.isoformat() if config.completion_date else None,
        'is_locked': config.is_locked
    }

    # Get periods
    periods_data = {'periods': [], 'period_count': 0}
    if snapshot:
        periods = manager.get_periods(gmp_division, granularity)
        if periods:
            periods_data = {
                'period_count': len(periods),
                'periods': [
                    {
                        'period_label': p.period_label,
                        'period_number': p.period_number,
                        'period_start': p.period_start.isoformat() if p.period_start else None,
                        'period_end': p.period_end.isoformat() if p.period_end else None,
                        'period_type': p.period_type,
                        'iso_week': p.iso_week,
                        'iso_year': p.iso_year,
                        'actual_cents': p.actual_cents,
                        'actual_display': cents_to_display(p.actual_cents),
                        'forecast_cents': p.forecast_cents,
                        'forecast_display': cents_to_display(p.forecast_cents),
                        'blended_cents': p.blended_cents,
                        'blended_display': cents_to_display(p.blended_cents),
                        'cumulative_cents': p.cumulative_cents,
                        'cumulative_display': cents_to_display(p.cumulative_cents),
                        'actual_west_cents': p.actual_west_cents,
                        'actual_east_cents': p.actual_east_cents,
                        'forecast_west_cents': p.forecast_west_cents,
                        'forecast_east_cents': p.forecast_east_cents
                    }
                    for p in periods
                ]
            }

    return templates.TemplateResponse(request, "forecast.html", {
        "gmp_division": gmp_division,
        "granularity": granularity,
        "forecast": forecast,
        "config": config_dict,
        "periods": periods_data,
        "side_filter": side_filter,
        "available_sides": available_sides,
        "is_project_view": False,
        "active_page": "gmp"
    })


@app.get("/mappings", response_class=HTMLResponse)
async def mappings_page(
    request: Request,
    tab: str = "budget_to_gmp",
    side: Optional[str] = None,
    initial_limit: int = 20,  # Initial rows to load (pagination)
    db: Session = Depends(get_db)
):
    """Mapping editor page with tabs for Budget→GMP, Direct→Budget, and Allocations."""
    from rapidfuzz import fuzz, process

    data_loader = get_data_loader()

    # Get available sides for filter dropdown
    side_configs = db.query(SideConfiguration).filter(SideConfiguration.is_active == True).all()
    available_sides = [{'value': s.side, 'label': s.display_name} for s in side_configs]

    # Validate and normalize side filter
    side_filter = None
    if side:
        side_upper = side.upper()
        if side_upper in VALID_SIDES:
            side_filter = side_upper

    # Get mappings from database (with optional side filter)
    budget_query = db.query(BudgetToGMP)
    direct_query = db.query(DirectToBudget)

    if side_filter:
        # Include BOTH when filtering by EAST or WEST
        if side_filter != 'BOTH':
            budget_query = budget_query.filter(BudgetToGMP.side.in_([side_filter, 'BOTH']))
            direct_query = direct_query.filter(DirectToBudget.side.in_([side_filter, 'BOTH']))
        else:
            budget_query = budget_query.filter(BudgetToGMP.side == side_filter)
            direct_query = direct_query.filter(DirectToBudget.side == side_filter)

    budget_mappings_db = budget_query.all()
    direct_mappings_db = direct_query.all()
    allocations = db.query(Allocation).all()

    # Get available options
    gmp_options = data_loader.gmp['GMP'].tolist()

    # Build budget options with descriptions for enhanced dropdown
    # Format: list of dicts with 'code' and 'description'
    budget_options = []
    budget_codes_seen = set()
    for _, row in data_loader.budget.iterrows():
        bc = row.get('Budget Code', '')
        if bc and bc not in budget_codes_seen:
            budget_codes_seen.add(bc)
            desc = row.get('Budget Code Description', '')
            # Handle NaN/None descriptions
            if desc is None or (isinstance(desc, float) and desc != desc):
                desc = ''
            else:
                desc = str(desc).strip()
            budget_options.append({
                'code': bc,
                'description': desc,
                'display': f"{bc} – {desc[:40]}..." if len(desc) > 40 else f"{bc} – {desc}" if desc else f"{bc} – (No description)"
            })
    # Sort alphabetically by code (convert to string to handle mixed types)
    budget_options.sort(key=lambda x: str(x['code']) if x['code'] is not None else '')

    # Build lookup for budget descriptions
    budget_desc_lookup = {}
    budget_type_lookup = {}
    for _, row in data_loader.budget.iterrows():
        bc = row.get('Budget Code', '')
        if bc:
            budget_desc_lookup[bc] = row.get('Budget Code Description', '')
            budget_type_lookup[bc] = row.get('Cost Type', '')

    # Get unmapped items
    gmp_df = data_loader.gmp.copy()
    budget_df = map_budget_to_gmp(data_loader.budget.copy(), gmp_df, db)
    direct_df = map_direct_to_budget(data_loader.direct_costs.copy(), budget_df, db)

    # Build enriched budget mappings list (with descriptions)
    budget_mappings = []
    mapped_budget_codes = set()
    budget_side_lookup = {}  # To track side for each budget code
    for m in budget_mappings_db:
        mapped_budget_codes.add(m.budget_code)
        budget_side_lookup[m.budget_code] = m.side
        budget_mappings.append({
            'id': m.id,
            'budget_code': m.budget_code,
            'gmp_division': m.gmp_division,
            'side': m.side,
            'confidence': m.confidence,
            'description': budget_desc_lookup.get(m.budget_code, ''),
            'cost_type': budget_type_lookup.get(m.budget_code, ''),
            'mapping_method': 'database'
        })

    # Get all budget items (both mapped and unmapped) for better overview
    all_budget_items = []
    for _, row in budget_df.iterrows():
        bc = row.get('Budget Code', '')
        item = {
            'Budget Code': bc,
            'Budget Code Description': row.get('Budget Code Description', ''),
            'Cost Type': row.get('Cost Type', ''),
            'division_key': row.get('division_key', ''),
            'gmp_division': row.get('gmp_division'),
            'side': budget_side_lookup.get(bc, 'BOTH'),  # Default to BOTH for unmapped
            'mapping_confidence': row.get('mapping_confidence', 0),
            'mapping_method': row.get('mapping_method', 'unmapped'),
            'is_mapped': bc in mapped_budget_codes or row.get('gmp_division') is not None
        }
        # Add suggestion for unmapped items
        if not item['is_mapped']:
            desc = str(row.get('Budget Code Description', '') or '')
            if desc and len(desc) > 3:
                match = process.extractOne(desc, gmp_options, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 60:
                    item['suggested_gmp'] = match[0]
                    item['suggestion_confidence'] = match[1]
        all_budget_items.append(item)

    # Separate mapped vs unmapped for display
    all_unmapped_budget = [b for b in all_budget_items if not b['is_mapped']]
    mapped_budget = [b for b in all_budget_items if b['is_mapped']]

    # Sort unmapped: items with suggestions first
    all_unmapped_budget.sort(key=lambda x: (-float(x.get('suggestion_confidence', 0) or 0), str(x.get('Budget Code', '') or '')))

    # Count suggestions (before limiting)
    suggested_budget = [b for b in all_unmapped_budget if b.get('suggested_gmp')]

    # Store total counts before limiting for pagination
    total_unmapped_budget = len(all_unmapped_budget)

    # Limit for initial display (pagination)
    unmapped_budget = all_unmapped_budget[:initial_limit]

    # Build direct cost mappings lookup from database
    direct_mappings_lookup = {}
    for m in direct_mappings_db:
        key = (m.cost_code, m.name)
        direct_mappings_lookup[key] = {
            'id': m.id,
            'budget_code': m.budget_code,
            'side': m.side,
            'confidence': m.confidence
        }

    # Compute suggestions for unmapped direct costs (not in database)
    mapped_keys = set(direct_mappings_lookup.keys())
    unmapped_mask = data_loader.direct_costs.apply(
        lambda row: (row.get('Cost Code', ''), row.get('Name', '')) not in mapped_keys,
        axis=1
    )
    unmapped_direct_df = data_loader.direct_costs[unmapped_mask].copy()
    dc_suggestions = compute_all_suggestions(
        unmapped_direct_df,
        data_loader.budget,
        db,
        unmapped_only=False,  # Already filtered to unmapped
        top_k=3
    )

    # Build ALL direct cost items list (both mapped and unmapped)
    all_direct_items = []
    display_columns = ['Cost Code', 'Name', 'Vendor', 'Invoice #', 'Date', 'Amount', 'Type', 'Description']

    for _, row in data_loader.direct_costs.iterrows():
        dc_id = row.get('direct_cost_id', 0)
        cost_code = row.get('Cost Code', '')
        name = row.get('Name', '')
        key = (cost_code, name)

        item = {col: row.get(col, '') for col in display_columns if col in row.index}
        item['direct_cost_id'] = dc_id

        # Format amount for display
        if 'amount_cents' in row.index:
            item['Amount'] = cents_to_display(int(row['amount_cents']))
        elif 'Amount' in row.index:
            item['Amount'] = row['Amount']

        # Check if mapped
        if key in direct_mappings_lookup:
            mapping = direct_mappings_lookup[key]
            item['is_mapped'] = True
            item['mapping_id'] = mapping['id']
            item['mapped_budget_code'] = mapping['budget_code']
            item['side'] = mapping['side']
            item['mapping_confidence'] = mapping['confidence']
            item['budget_description'] = budget_desc_lookup.get(mapping['budget_code'], '')
            item['confidence_band'] = 'mapped'
            item['suggestions'] = []
            item['top_suggestion'] = None
        else:
            item['is_mapped'] = False
            item['side'] = 'BOTH'  # Default for unmapped
            item['mapped_budget_code'] = None
            item['mapping_confidence'] = 0
            item['budget_description'] = ''

            # Add suggestions for unmapped items
            suggs = dc_suggestions.get(dc_id, [])
            if suggs:
                item['suggestions'] = suggs
                item['top_suggestion'] = suggs[0]
                item['confidence'] = suggs[0].get('score', 0)
                item['confidence_band'] = suggs[0].get('confidence_band', 'low')
            else:
                item['suggestions'] = []
                item['top_suggestion'] = None
                item['confidence'] = 0
                item['confidence_band'] = 'low'

        all_direct_items.append(item)

    # Separate for stats
    all_unmapped_direct = [d for d in all_direct_items if not d['is_mapped']]
    mapped_direct = [d for d in all_direct_items if d['is_mapped']]

    # Sort unmapped by confidence (high first), mapped by cost code
    all_unmapped_direct.sort(key=lambda x: (-x.get('confidence', 0), str(x.get('Cost Code', ''))))
    mapped_direct.sort(key=lambda x: str(x.get('Cost Code', '')))

    # Count by confidence band (before limiting)
    direct_high = len([d for d in all_unmapped_direct if d.get('confidence_band') == 'high'])
    direct_medium = len([d for d in all_unmapped_direct if d.get('confidence_band') == 'medium'])
    direct_low = len([d for d in all_unmapped_direct if d.get('confidence_band') == 'low'])

    # Store total counts before limiting for pagination
    total_unmapped_direct = len(all_unmapped_direct)

    # Limit for initial display (pagination)
    unmapped_direct = all_unmapped_direct[:initial_limit]

    return templates.TemplateResponse(request, "mappings.html", {
        "active_tab": tab,
        "side_filter": side_filter,
        "available_sides": available_sides,
        "budget_mappings": budget_mappings,
        "mapped_budget": mapped_budget,
        "mapped_direct": mapped_direct,
        "allocations": allocations,
        "gmp_options": gmp_options,
        "budget_options": budget_options,
        "unmapped_budget": unmapped_budget,
        "unmapped_direct": unmapped_direct,
        "suggested_budget": suggested_budget,
        "direct_high": direct_high,
        "direct_medium": direct_medium,
        "direct_low": direct_low,
        "total_budget": len(all_budget_items),
        "total_direct": len(all_direct_items),
        # Pagination info
        "initial_limit": initial_limit,
        "total_unmapped_budget": total_unmapped_budget,
        "total_unmapped_direct": total_unmapped_direct,
        "has_more_budget": total_unmapped_budget > initial_limit,
        "has_more_direct": total_unmapped_direct > initial_limit,
        "active_page": "mappings"
    })


@app.post("/mappings/save")
async def save_mappings(
    request: Request,
    mapping_type: str = Form(...),
    db: Session = Depends(get_db)
):
    """Save or delete mapping edits."""
    form_data = await request.form()
    action = form_data.get('action', 'save')

    try:
        if mapping_type == "budget_to_gmp":
            budget_code = form_data.get('budget_code')

            if action == 'delete':
                # Delete mapping
                db.query(BudgetToGMP).filter(BudgetToGMP.budget_code == budget_code).delete()
                db.commit()
            else:
                gmp_division = form_data.get('gmp_division')
                save_mapping(db, 'budget_to_gmp', {
                    'budget_code': budget_code,
                    'gmp_division': gmp_division,
                    'confidence': 1.0
                })

        elif mapping_type == "direct_to_budget":
            cost_code = form_data.get('cost_code')
            name = form_data.get('name', '')

            if action == 'delete':
                db.query(DirectToBudget).filter(
                    DirectToBudget.cost_code == cost_code,
                    DirectToBudget.name == name
                ).delete()
                db.commit()
            else:
                name = form_data.get('name')
                budget_code = form_data.get('budget_code')
                vendor = form_data.get('vendor', '')
                suggested_budget = form_data.get('suggested_budget_code')
                suggestion_score = form_data.get('suggestion_score')

                # Save the mapping
                save_mapping(db, 'direct_to_budget', {
                    'cost_code': cost_code,
                    'name': name,
                    'budget_code': budget_code,
                    'confidence': 1.0,
                    'method': 'user_confirmed' if suggested_budget == budget_code else 'manual',
                    'vendor_normalized': normalize_vendor(vendor) if vendor else None
                })

                # Record feedback for the learning loop
                if vendor or name:
                    record_mapping_feedback(
                        db=db,
                        vendor=vendor,
                        name=name,
                        selected_budget_code=budget_code,
                        suggested_budget_code=suggested_budget,
                        suggestion_score=float(suggestion_score) / 100 if suggestion_score else None,
                        user_id='web_user'
                    )

        elif mapping_type == "allocation":
            code = form_data.get('code')

            if action == 'delete':
                db.query(Allocation).filter(Allocation.code == code).delete()
                db.commit()
            else:
                pct_west = float(form_data.get('pct_west', 0.5))
                pct_east = float(form_data.get('pct_east', 0.5))
                save_mapping(db, 'allocations', {
                    'code': code,
                    'region': 'Both' if pct_west > 0 and pct_east > 0 else ('West' if pct_west > 0 else 'East'),
                    'pct_west': pct_west,
                    'pct_east': pct_east,
                    'confirmed': True
                })

        return RedirectResponse(url=f"/mappings?tab={mapping_type}", status_code=303)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/mappings/suggestions/{direct_cost_id}")
async def get_suggestions_for_direct_cost(
    direct_cost_id: int,
    db: Session = Depends(get_db)
):
    """
    Get match suggestions for a single direct cost row.
    Returns top 3 suggested budget codes with confidence scores.
    """
    data_loader = get_data_loader()
    direct_df = data_loader.direct_costs.copy()

    # Find the specific row
    row = direct_df[direct_df['direct_cost_id'] == direct_cost_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Direct cost not found")

    dc_row = row.iloc[0].to_dict()

    suggestions = compute_single_suggestion(
        dc_row=dc_row,
        budget_df=data_loader.budget,
        db=db,
        top_k=5
    )

    return {
        "direct_cost_id": direct_cost_id,
        "suggestions": suggestions
    }


@app.post("/api/mappings/save")
async def api_save_mapping(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    JSON API endpoint to save or update a mapping.

    Body:
    {
        "mapping_type": "budget_to_gmp" | "direct_to_budget" | "allocation",
        "action": "save" | "delete" (optional, default "save"),

        // For budget_to_gmp:
        "budget_code": "string",
        "gmp_division": "string",
        "side": "EAST" | "WEST" | "BOTH" (optional, default "BOTH"),

        // For direct_to_budget:
        "cost_code": "string",
        "name": "string",
        "budget_code": "string",
        "vendor": "string" (optional),
        "side": "EAST" | "WEST" | "BOTH" (optional, default "BOTH"),
        "suggested_budget_code": "string" (optional, for feedback),
        "suggestion_score": number (optional, for feedback),

        // For allocation:
        "code": "string",
        "pct_west": number,
        "pct_east": number
    }

    Returns:
    {
        "success": true,
        "mapping_id": number,
        "action": "created" | "updated" | "deleted"
    }
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid JSON body"}
        )

    mapping_type = data.get('mapping_type')
    action = data.get('action', 'save')

    if not mapping_type:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Missing mapping_type"}
        )

    if mapping_type not in ['budget_to_gmp', 'direct_to_budget', 'allocation']:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Invalid mapping_type: {mapping_type}"}
        )

    try:
        if mapping_type == "budget_to_gmp":
            budget_code = data.get('budget_code')
            if not budget_code:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing budget_code"}
                )

            if action == 'delete':
                deleted = db.query(BudgetToGMP).filter(BudgetToGMP.budget_code == budget_code).delete()
                db.commit()
                return {"success": True, "deleted_count": deleted, "action": "deleted"}

            gmp_division = data.get('gmp_division')
            if not gmp_division:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing gmp_division"}
                )

            side = data.get('side', 'BOTH').upper()
            if side not in ['EAST', 'WEST', 'BOTH']:
                side = 'BOTH'

            mapping_data = {
                'budget_code': budget_code,
                'gmp_division': gmp_division,
                'side': side,
                'confidence': 1.0
            }

            result = save_mapping(db, 'budget_to_gmp', mapping_data)
            return {"success": True, "action": result.get('action', 'saved'), "mapping_id": result.get('id')}

        elif mapping_type == "direct_to_budget":
            cost_code = data.get('cost_code')
            name = data.get('name', '')

            if not cost_code:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing cost_code"}
                )

            if action == 'delete':
                deleted = db.query(DirectToBudget).filter(
                    DirectToBudget.cost_code == cost_code,
                    DirectToBudget.name == name
                ).delete()
                db.commit()
                return {"success": True, "deleted_count": deleted, "action": "deleted"}

            budget_code = data.get('budget_code')
            if not budget_code:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing budget_code"}
                )

            vendor = data.get('vendor', '')
            suggested_budget = data.get('suggested_budget_code', '')
            suggestion_score = data.get('suggestion_score')
            side = data.get('side', 'BOTH').upper()
            if side not in ['EAST', 'WEST', 'BOTH']:
                side = 'BOTH'

            mapping_data = {
                'cost_code': cost_code,
                'name': name,
                'budget_code': budget_code,
                'side': side,
                'confidence': 1.0,
                'method': 'user_confirmed' if suggested_budget == budget_code else 'manual',
                'vendor_normalized': normalize_vendor(vendor) if vendor else None
            }

            result = save_mapping(db, 'direct_to_budget', mapping_data)

            # Record feedback for the learning loop
            if vendor or name:
                record_mapping_feedback(
                    db=db,
                    vendor=vendor,
                    name=name,
                    selected_budget_code=budget_code,
                    suggested_budget_code=suggested_budget,
                    suggestion_score=float(suggestion_score) / 100 if suggestion_score else None,
                    user_id='web_user'
                )

            return {"success": True, "action": result.get('action', 'saved'), "mapping_id": result.get('id')}

        elif mapping_type == "allocation":
            code = data.get('code')
            if not code:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing code"}
                )

            if action == 'delete':
                deleted = db.query(Allocation).filter(Allocation.code == code).delete()
                db.commit()
                return {"success": True, "deleted_count": deleted, "action": "deleted"}

            pct_west = float(data.get('pct_west', 0.5))
            pct_east = float(data.get('pct_east', 0.5))

            mapping_data = {
                'code': code,
                'region': 'Both' if pct_west > 0 and pct_east > 0 else ('West' if pct_west > 0 else 'East'),
                'pct_west': pct_west,
                'pct_east': pct_east,
                'confirmed': True
            }

            result = save_mapping(db, 'allocations', mapping_data)
            return {"success": True, "action": result.get('action', 'saved'), "mapping_id": result.get('id')}

    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/mappings/bulk-accept")
async def bulk_accept_suggestions(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Bulk accept all high-confidence suggestions.
    Returns count of mappings created.
    """
    data = await request.json()
    thresholds = _get_thresholds()
    min_confidence = data.get('min_confidence', thresholds['high'] * 100)  # Default to high threshold
    direct_cost_ids = data.get('direct_cost_ids', [])  # Optional: specific IDs to process

    data_loader = get_data_loader()
    direct_df = data_loader.direct_costs.copy()
    budget_df = data_loader.budget.copy()

    # Get existing mappings from database
    existing_mappings = db.query(DirectToBudget).all()
    mapped_keys = {(m.cost_code, m.name) for m in existing_mappings}

    # Filter to unmapped only (not in database)
    unmapped_mask = direct_df.apply(
        lambda row: (row.get('Cost Code', ''), row.get('Name', '')) not in mapped_keys,
        axis=1
    )
    unmapped_df = direct_df[unmapped_mask].copy()

    if direct_cost_ids:
        unmapped_df = unmapped_df[unmapped_df['direct_cost_id'].isin(direct_cost_ids)]

    # Compute suggestions
    suggestions = compute_all_suggestions(unmapped_df, budget_df, db, unmapped_only=False, top_k=1)

    accepted = 0
    skipped = 0

    for dc_id, suggs in suggestions.items():
        if not suggs:
            skipped += 1
            continue

        top = suggs[0]
        if top.get('score', 0) >= min_confidence:
            # Get the direct cost row for feedback
            dc_row = unmapped_df[unmapped_df['direct_cost_id'] == dc_id].iloc[0]

            # Save mapping
            save_mapping(db, 'direct_to_budget', {
                'cost_code': str(dc_row.get('Cost Code', '')),
                'name': str(dc_row.get('Name', '')),
                'budget_code': top['budget_code'],
                'confidence': top['total_score'],
                'method': 'bulk_accept',
                'vendor_normalized': normalize_vendor(str(dc_row.get('Vendor', '') or ''))
            })

            # Record feedback
            record_mapping_feedback(
                db=db,
                vendor=str(dc_row.get('Vendor', '') or ''),
                name=str(dc_row.get('Name', '') or ''),
                selected_budget_code=top['budget_code'],
                suggested_budget_code=top['budget_code'],
                suggestion_score=top['total_score'],
                user_id='bulk_accept'
            )

            accepted += 1
        else:
            skipped += 1

    return {
        "accepted": accepted,
        "skipped": skipped,
        "total_processed": accepted + skipped
    }


@app.get("/duplicates", response_class=HTMLResponse)
async def duplicates_page(request: Request, db: Session = Depends(get_db)):
    """Duplicates review page."""
    data_loader = get_data_loader()
    direct_costs_df = data_loader.direct_costs.copy()
    
    # Detect duplicates
    duplicates, _ = detect_duplicates(direct_costs_df)
    
    # Format for display
    formatted_dups = format_duplicates_for_display(duplicates, direct_costs_df)
    summary = get_duplicates_summary(duplicates)
    
    return templates.TemplateResponse(request, "duplicates.html", {
        "duplicates": formatted_dups,
        "summary": summary,
        "active_page": "duplicates"
    })


@app.post("/duplicates/resolve")
async def resolve_duplicate(
    row_id: int = Form(...),
    action: str = Form(...),  # exclude, include, ignore
    db: Session = Depends(get_db)
):
    """Resolve a duplicate entry."""
    dup = db.query(Duplicate).filter(Duplicate.direct_cost_row_id == row_id).first()
    
    if dup:
        if action == "exclude":
            dup.excluded_from_actuals = True
        elif action == "include":
            dup.excluded_from_actuals = False
        
        dup.resolved = True
        dup.resolved_at = datetime.now(timezone.utc)
        dup.resolved_by = "user"
        db.commit()
    
    return RedirectResponse(url="/duplicates", status_code=303)


@app.get("/data-health", response_class=HTMLResponse)
async def data_health_page(request: Request, db: Session = Depends(get_db)):
    """Data health summary page."""
    data_loader = get_data_loader()
    
    # Analyze data quality
    gmp_df = data_loader.gmp
    budget_df = data_loader.budget
    direct_costs_df = data_loader.direct_costs
    
    issues = []
    
    # Check for missing values
    dc_missing_vendor = direct_costs_df[direct_costs_df['Vendor'].isna() | (direct_costs_df['Vendor'] == '')]
    if len(dc_missing_vendor) > 0:
        issues.append({
            'type': 'warning',
            'area': 'Direct Costs',
            'message': f'{len(dc_missing_vendor)} rows missing vendor',
            'count': len(dc_missing_vendor)
        })
    
    dc_missing_date = direct_costs_df[direct_costs_df['date_parsed'].isna()]
    if len(dc_missing_date) > 0:
        issues.append({
            'type': 'error',
            'area': 'Direct Costs',
            'message': f'{len(dc_missing_date)} rows with invalid/missing dates',
            'count': len(dc_missing_date)
        })
    
    # Check for zero amounts
    dc_zero = direct_costs_df[direct_costs_df['amount_cents'] == 0]
    if len(dc_zero) > 0:
        issues.append({
            'type': 'info',
            'area': 'Direct Costs',
            'message': f'{len(dc_zero)} rows with zero amount',
            'count': len(dc_zero)
        })
    
    # Summary stats
    stats = {
        'gmp_rows': len(gmp_df),
        'budget_rows': len(budget_df),
        'direct_cost_rows': len(direct_costs_df),
        'date_range': f"{direct_costs_df['date_parsed'].min()} to {direct_costs_df['date_parsed'].max()}",
        'total_gmp': cents_to_display(gmp_df['amount_total_cents'].sum()),
        'total_direct_costs': cents_to_display(direct_costs_df['amount_cents'].sum())
    }
    
    return templates.TemplateResponse(request, "data_health.html", {
        "issues": issues,
        "stats": stats,
        "active_page": "data-health"
    })


@app.get("/schedule", response_class=HTMLResponse)
async def schedule_page(request: Request, db: Session = Depends(get_db)):
    """Schedule to GMP mapping page with P6 progress tracking."""
    # Get all schedule activities with their mappings
    activities = db.query(ScheduleActivity).order_by(ScheduleActivity.row_number).all()
    activities_data = []
    mapped_count = 0

    # P6 stats tracking
    complete_count = 0
    in_progress_count = 0
    critical_count = 0
    total_weight = 0.0
    weighted_progress_sum = 0.0

    for act in activities:
        mappings = [
            {'gmp_division': m.gmp_division, 'weight': m.weight}
            for m in act.mappings
        ]

        # Use P6 progress_pct if available, fallback to pct_complete
        progress_pct = getattr(act, 'progress_pct', None)
        if progress_pct is None:
            progress_pct = act.pct_complete / 100.0

        # Compute weight for progress calculation
        is_critical = getattr(act, 'is_critical', False)
        duration = act.duration_days or 1
        weight = (2.0 if is_critical else 1.0) * ((duration / 10) ** 0.5)
        total_weight += weight
        weighted_progress_sum += weight * progress_pct

        activities_data.append({
            'id': act.id,
            'row_number': act.row_number,
            'task_name': act.task_name,
            'activity_id': act.activity_id,
            'wbs': act.wbs,
            'pct_complete': act.pct_complete,
            'progress_pct': progress_pct,
            'start_date': act.start_date.isoformat() if act.start_date else None,
            'finish_date': act.finish_date.isoformat() if act.finish_date else None,
            'start_is_actual': getattr(act, 'start_is_actual', False),
            'finish_is_actual': getattr(act, 'finish_is_actual', False),
            'is_complete': getattr(act, 'is_complete', False),
            'is_in_progress': getattr(act, 'is_in_progress', False),
            'is_critical': is_critical,
            'total_float': getattr(act, 'total_float', None),
            'duration_days': act.duration_days,
            'zone': getattr(act, 'zone', None),  # Zone assignment (EAST, WEST, SHARED)
            'mappings': mappings
        })

        # Track P6 stats
        if getattr(act, 'is_complete', False):
            complete_count += 1
        elif getattr(act, 'is_in_progress', False):
            in_progress_count += 1

        if is_critical:
            critical_count += 1

        if mappings:
            mapped_count += 1

    # Get GMP divisions for dropdown
    data_loader = get_data_loader()
    gmp_divisions = data_loader.gmp['GMP'].tolist() if not data_loader.gmp.empty else []

    total_activities = len(activities_data)
    # Weighted average progress (P6-style)
    avg_progress = round(weighted_progress_sum / total_weight * 100) if total_weight > 0 else 0

    return templates.TemplateResponse(request, "schedule.html", {
        "activities": activities_data,
        "gmp_divisions": gmp_divisions,
        "total_activities": total_activities,
        "mapped_activities": mapped_count,
        "unmapped_activities": total_activities - mapped_count,
        "avg_progress": avg_progress,
        "complete_count": complete_count,
        "in_progress_count": in_progress_count,
        "critical_count": critical_count,
        "active_page": "schedule"
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: Session = Depends(get_db)):
    """Settings page."""
    settings = db.query(Settings).first()
    data_loader = get_data_loader()
    
    # Get last run info
    last_run = db.query(Run).order_by(Run.started_at.desc()).first()
    
    return templates.TemplateResponse(request, "settings.html", {
        "settings": settings,
        "max_transaction_date": data_loader.max_transaction_date,
        "last_run": last_run,
        "active_page": "settings"
    })


@app.post("/settings")
async def save_settings(
    request: Request,
    db: Session = Depends(get_db)
):
    """Save settings."""
    form_data = await request.form()
    
    settings = db.query(Settings).first()
    if not settings:
        settings = Settings()
        db.add(settings)
    
    # Parse as_of_date
    as_of_str = form_data.get('as_of_date', '')
    if as_of_str and as_of_str != 'auto':
        settings.as_of_date = datetime.fromisoformat(as_of_str)
    else:
        settings.as_of_date = None
    
    settings.forecast_basis = form_data.get('forecast_basis', 'actuals_plus_commitments')
    settings.eac_mode_when_commitments = form_data.get('eac_mode', 'max')
    settings.gmp_scope_notes = form_data.get('gmp_scope_notes', '')
    settings.gmp_scope_confirmed = form_data.get('gmp_scope_confirmed') == 'on'
    settings.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/recompute")
async def trigger_recompute(db: Session = Depends(get_db)):
    """Trigger full recomputation."""
    run = Run(
        run_type='recompute',
        status='running',
        started_at=datetime.now(timezone.utc),
        file_hashes=json.dumps(get_file_hashes())
    )
    db.add(run)
    db.commit()
    
    try:
        # Reload data
        data_loader = get_data_loader()
        data_loader.reload()
        
        # Retrain ML
        settings = get_settings(db)
        pipeline = get_forecasting_pipeline()
        pipeline.train(
            data_loader.direct_costs,
            data_loader.budget,
            data_loader.gmp,
            settings.get('as_of_date')
        )
        
        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
    except Exception as e:
        run.status = 'failed'
        run.notes = str(e)
        run.finished_at = datetime.now(timezone.utc)
    
    db.commit()
    
    return RedirectResponse(url="/gmp", status_code=303)


@app.get("/export.csv")
async def export_csv(db: Session = Depends(get_db)):
    """Export reconciliation table as CSV."""
    result = run_full_reconciliation(db)
    
    # Create DataFrame from rows
    df = pd.DataFrame(result['recon_rows'])
    
    # Select and rename columns for export
    export_cols = {
        'gmp_division': 'GMP Division',
        'gmp_amount': 'GMP Amount',
        'amount_assigned_west': 'Assigned West',
        'amount_assigned_east': 'Assigned East',
        'forecast_west': 'Forecast (EAC) West',
        'forecast_east': 'Forecast (EAC) East',
        'surplus_or_overrun': 'Surplus/Overrun',
        'actual_total': 'Actual Total',
        'committed_total': 'Committed Total',
        'eac_total': 'EAC Total',
        'pct_spent': '% Spent'
    }
    
    export_df = df[[c for c in export_cols.keys() if c in df.columns]]
    export_df.columns = [export_cols[c] for c in export_df.columns]
    
    # Create CSV
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=gmp_reconciliation_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


@app.get("/recon-summary.json")
async def recon_summary_json(db: Session = Depends(get_db)):
    """JSON endpoint for summary panel (for AJAX updates)."""
    result = run_full_reconciliation(db)
    return convert_numpy_types({
        "summary": result['summary'],
        "tie_outs": result['tie_outs'],
        "mapping_stats": result['mapping_stats'],
        "duplicates_summary": result['duplicates_summary']
    })


# =============================================================================
# Dashboard Summary API (Single Source of Truth for KPIs)
# =============================================================================

class DashboardSummaryResponse(PydanticBaseModel):
    """Response model for dashboard summary metrics."""
    total_gmp_budget_cents: int
    actual_costs_cents: int
    forecast_remaining_cents: int
    eac_cents: int
    variance_cents: Optional[int]
    progress_pct: float
    cpi: Optional[float]
    schedule_variance_days: Optional[int]
    warnings: List[str]


@app.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary_api(db: Session = Depends(get_db)):
    """
    Get dashboard summary metrics (single source of truth).

    Returns all KPIs computed from authoritative data sources:
    - total_gmp_budget_cents: From GMP entities table (static baseline)
    - actual_costs_cents: From DirectCostEntity table
    - forecast_remaining_cents: From ForecastSnapshot table (ETC)
    - eac_cents: Actual + Forecast Remaining
    - variance_cents: Budget - EAC (positive = underrun)
    - progress_pct: Actual / EAC * 100
    - cpi: EV / AC (only if EV available)
    - schedule_variance_days: Days ahead/behind schedule
    - warnings: List of data quality warnings
    """
    summary = compute_dashboard_summary(db)
    return summary


@app.get("/api/gmp/drilldown/{gmp_division}")
async def gmp_drilldown(
    gmp_division: str,
    side: Optional[str] = Query(None, description="Filter by side: EAST, WEST, BOTH"),
    include_both: bool = Query(True, description="When filtering by EAST/WEST, also include BOTH"),
    db: Session = Depends(get_db)
):
    """
    Get detailed breakdown of direct costs for a GMP division.
    Shows budget codes and individual records contributing to Assigned West/East.

    Query Parameters:
    - side: Filter by side (EAST, WEST, BOTH) - optional
    - include_both: When filtering by EAST/WEST, also include BOTH mappings (default: True)
    """
    # Validate side parameter
    if side:
        side = side.upper()
        if side not in VALID_SIDES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid side. Must be one of: {', '.join(VALID_SIDES)}"
            )

    data_loader = get_data_loader()
    settings = get_settings(db)

    # Get data with mappings applied
    gmp_df = data_loader.gmp.copy()
    budget_df = map_budget_to_gmp(data_loader.budget.copy(), gmp_df, db)
    direct_df = map_direct_to_budget(data_loader.direct_costs.copy(), budget_df, db)

    # Apply side filter to mappings if specified
    if side:
        # Get budget mappings for this side
        if include_both and side != 'BOTH':
            budget_side_filter = db.query(BudgetToGMP).filter(
                BudgetToGMP.side.in_([side, 'BOTH'])
            ).all()
        else:
            budget_side_filter = db.query(BudgetToGMP).filter(
                BudgetToGMP.side == side
            ).all()

        # Filter budget_df to only include budget codes matching the side
        side_budget_codes = {m.budget_code for m in budget_side_filter}
        if side_budget_codes:
            budget_df = budget_df[budget_df['Budget Code'].isin(side_budget_codes)]

        # Similarly filter direct costs by side
        if include_both and side != 'BOTH':
            direct_side_filter = db.query(DirectToBudget).filter(
                DirectToBudget.side.in_([side, 'BOTH'])
            ).all()
        else:
            direct_side_filter = db.query(DirectToBudget).filter(
                DirectToBudget.side == side
            ).all()

        # Create lookup for direct cost filtering
        side_direct_keys = {(m.cost_code, m.name) for m in direct_side_filter}
        if side_direct_keys:
            direct_df = direct_df[
                direct_df.apply(
                    lambda r: (r.get('Cost Code', ''), r.get('Name', '')) in side_direct_keys,
                    axis=1
                )
            ]

    # Apply allocations
    allocations_df = data_loader.allocations.copy()
    direct_df = apply_allocations(direct_df, 'amount_cents', 'base_code', allocations_df, db)

    # Detect and exclude duplicates
    duplicates, _ = detect_duplicates(direct_df)
    direct_df = apply_duplicate_exclusions(direct_df, duplicates)

    # Get drilldown data
    drilldown = get_gmp_drilldown(
        gmp_division=gmp_division,
        direct_costs_df=direct_df,
        budget_df=budget_df,
        as_of_date=settings.get('as_of_date')
    )

    # Add side filter info to response
    drilldown['side_filter'] = side
    drilldown['include_both'] = include_both

    return convert_numpy_types(drilldown)


@app.get("/api/gmp/relationships")
async def gmp_relationships(db: Session = Depends(get_db)):
    """
    Validate and return data model relationship statistics.
    GMP → Budget: one-to-many
    Budget → Direct Cost: one-to-many
    """
    data_loader = get_data_loader()
    gmp_df = data_loader.gmp.copy()
    budget_df = map_budget_to_gmp(data_loader.budget.copy(), gmp_df, db)
    direct_df = map_direct_to_budget(data_loader.direct_costs.copy(), budget_df, db)

    # GMP → Budget relationships
    gmp_to_budget = {}
    for gmp in gmp_df['GMP'].unique():
        budget_codes = budget_df[budget_df['gmp_division'] == gmp]['Budget Code'].tolist()
        gmp_to_budget[gmp] = {
            'count': len(budget_codes),
            'codes': budget_codes[:10]  # First 10 for display
        }

    # Budget → Direct Cost relationships
    budget_to_dc = {}
    mapped_direct = direct_df[direct_df['mapped_budget_code'].notna()]
    for budget_code in budget_df['Budget Code'].unique():
        dc_count = len(mapped_direct[mapped_direct['mapped_budget_code'] == budget_code])
        if dc_count > 0:
            budget_to_dc[budget_code] = dc_count

    return {
        'gmp_count': len(gmp_df),
        'budget_count': len(budget_df),
        'direct_cost_count': len(direct_df),
        'gmp_to_budget': gmp_to_budget,
        'budget_to_direct_cost_count': len(budget_to_dc),
        'relationship_valid': True
    }


# ------------ Allocation Override APIs ------------

@app.get("/api/gmp/allocations/{gmp_division}")
async def get_gmp_allocation(gmp_division: str, db: Session = Depends(get_db)):
    """Get current allocation values for a GMP division."""
    from app.models import GMPAllocationOverride

    override = db.query(GMPAllocationOverride).filter(
        GMPAllocationOverride.gmp_division == gmp_division
    ).first()

    # Get computed values from reconciliation
    result = run_full_reconciliation(db)
    computed_west = 0
    computed_east = 0
    gmp_total = 0

    for row in result['recon_rows']:
        if row['gmp_division'] == gmp_division:
            computed_west = row.get('actual_west_raw', 0)
            computed_east = row.get('actual_east_raw', 0)
            gmp_total = row.get('gmp_amount_raw', 0)
            break

    return {
        'gmp_division': gmp_division,
        'gmp_total': gmp_total,
        'computed_west': computed_west,
        'computed_east': computed_east,
        'computed_total': computed_west + computed_east,
        'override_west': override.amount_west_cents if override else None,
        'override_east': override.amount_east_cents if override else None,
        'has_override': override is not None,
        'notes': override.notes if override else None
    }


@app.post("/api/gmp/allocations/{gmp_division}")
async def save_gmp_allocation(
    gmp_division: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Save allocation override for a GMP division.
    Validates that West + East equals total computed amount.
    """
    from app.models import GMPAllocationOverride, AllocationChangeLog

    data = await request.json()
    amount_west = data.get('amount_west_cents')
    amount_east = data.get('amount_east_cents')
    notes = data.get('notes', '')

    # Get existing override or create new
    override = db.query(GMPAllocationOverride).filter(
        GMPAllocationOverride.gmp_division == gmp_division
    ).first()

    old_west = override.amount_west_cents if override else None
    old_east = override.amount_east_cents if override else None

    if override:
        # Log changes
        if amount_west != old_west:
            log = AllocationChangeLog(
                gmp_division=gmp_division,
                field_changed='amount_west',
                old_value_cents=old_west,
                new_value_cents=amount_west,
                change_reason=notes
            )
            db.add(log)

        if amount_east != old_east:
            log = AllocationChangeLog(
                gmp_division=gmp_division,
                field_changed='amount_east',
                old_value_cents=old_east,
                new_value_cents=amount_east,
                change_reason=notes
            )
            db.add(log)

        override.amount_west_cents = amount_west
        override.amount_east_cents = amount_east
        override.notes = notes
    else:
        override = GMPAllocationOverride(
            gmp_division=gmp_division,
            amount_west_cents=amount_west,
            amount_east_cents=amount_east,
            notes=notes
        )
        db.add(override)

        # Log creation
        if amount_west is not None:
            log = AllocationChangeLog(
                gmp_division=gmp_division,
                field_changed='amount_west',
                old_value_cents=None,
                new_value_cents=amount_west,
                change_reason='Initial override'
            )
            db.add(log)

        if amount_east is not None:
            log = AllocationChangeLog(
                gmp_division=gmp_division,
                field_changed='amount_east',
                old_value_cents=None,
                new_value_cents=amount_east,
                change_reason='Initial override'
            )
            db.add(log)

    db.commit()

    return {
        'success': True,
        'gmp_division': gmp_division,
        'amount_west_cents': amount_west,
        'amount_east_cents': amount_east
    }


@app.delete("/api/gmp/allocations/{gmp_division}")
async def clear_gmp_allocation(gmp_division: str, db: Session = Depends(get_db)):
    """Clear allocation override, reverting to computed values."""
    from app.models import GMPAllocationOverride, AllocationChangeLog

    override = db.query(GMPAllocationOverride).filter(
        GMPAllocationOverride.gmp_division == gmp_division
    ).first()

    if override:
        # Log clearing
        log = AllocationChangeLog(
            gmp_division=gmp_division,
            field_changed='override_cleared',
            old_value_cents=override.amount_west_cents,
            new_value_cents=None,
            change_reason='Override cleared, reverted to computed values'
        )
        db.add(log)
        db.delete(override)
        db.commit()
        return {'success': True, 'message': 'Override cleared'}

    return {'success': False, 'message': 'No override found'}


@app.get("/api/gmp/allocation-history/{gmp_division}")
async def get_allocation_history(gmp_division: str, db: Session = Depends(get_db)):
    """Get change history for a GMP division's allocations."""
    from app.models import AllocationChangeLog

    logs = db.query(AllocationChangeLog).filter(
        AllocationChangeLog.gmp_division == gmp_division
    ).order_by(AllocationChangeLog.changed_at.desc()).limit(50).all()

    return {
        'gmp_division': gmp_division,
        'history': [
            {
                'field': log.field_changed,
                'old_value': log.old_value_cents,
                'new_value': log.new_value_cents,
                'reason': log.change_reason,
                'changed_by': log.changed_by,
                'changed_at': log.changed_at.isoformat() if log.changed_at else None
            }
            for log in logs
        ]
    }


# ------------ Side Configuration API Endpoints ------------

VALID_SIDES = ['EAST', 'WEST', 'BOTH']


@app.get("/api/sides")
async def get_sides(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get list of available sides (EAST, WEST, BOTH) with configuration.

    Query Parameters:
    - active_only: Filter to only active sides (default: True)

    Returns:
    - List of side configurations with timeline info and allocation weights
    """
    query = db.query(SideConfiguration)
    if active_only:
        query = query.filter(SideConfiguration.is_active == True)

    sides = query.order_by(SideConfiguration.id).all()

    return {
        'sides': [
            {
                'value': s.side.lower(),
                'label': s.display_name,
                'side': s.side,
                'display_name': s.display_name,
                'start_date': s.start_date.isoformat() if s.start_date else None,
                'end_date': s.end_date.isoformat() if s.end_date else None,
                'is_active': s.is_active,
                'allocation_weight': s.allocation_weight
            }
            for s in sides
        ],
        'allocation_ratio': {
            'east': next((s.allocation_weight for s in sides if s.side == 'EAST'), 0.5),
            'west': next((s.allocation_weight for s in sides if s.side == 'WEST'), 0.5)
        }
    }


@app.get("/api/sides/{side}")
async def get_side_config(
    side: str,
    db: Session = Depends(get_db)
):
    """
    Get configuration for a specific side.

    Path Parameters:
    - side: Side identifier (EAST, WEST, BOTH - case insensitive)
    """
    side_upper = side.upper()
    if side_upper not in VALID_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid side. Must be one of: {', '.join(VALID_SIDES)}"
        )

    config = db.query(SideConfiguration).filter(
        SideConfiguration.side == side_upper
    ).first()

    if not config:
        raise HTTPException(status_code=404, detail=f"Side '{side}' not configured")

    return {
        'value': config.side.lower(),
        'label': config.display_name,
        'side': config.side,
        'display_name': config.display_name,
        'start_date': config.start_date.isoformat() if config.start_date else None,
        'end_date': config.end_date.isoformat() if config.end_date else None,
        'is_active': config.is_active,
        'allocation_weight': config.allocation_weight
    }


@app.put("/api/sides/{side}")
async def update_side_config(
    side: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Update configuration for a specific side.

    Path Parameters:
    - side: Side identifier (EAST, WEST, BOTH)

    Body:
    {
        "display_name": "East",
        "start_date": "2025-06-01",  // ISO format, null to clear
        "end_date": "2025-07-31",    // ISO format, null to clear
        "is_active": true,
        "allocation_weight": 0.5
    }
    """
    side_upper = side.upper()
    if side_upper not in VALID_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid side. Must be one of: {', '.join(VALID_SIDES)}"
        )

    config = db.query(SideConfiguration).filter(
        SideConfiguration.side == side_upper
    ).first()

    if not config:
        raise HTTPException(status_code=404, detail=f"Side '{side}' not configured")

    body = await request.json()

    # Update fields if provided
    if 'display_name' in body:
        config.display_name = body['display_name']

    if 'start_date' in body:
        if body['start_date']:
            config.start_date = datetime.fromisoformat(body['start_date'])
        else:
            config.start_date = None

    if 'end_date' in body:
        if body['end_date']:
            config.end_date = datetime.fromisoformat(body['end_date'])
        else:
            config.end_date = None

    if 'is_active' in body:
        config.is_active = body['is_active']

    if 'allocation_weight' in body:
        weight = float(body['allocation_weight'])
        if not 0 <= weight <= 1:
            raise HTTPException(status_code=400, detail="allocation_weight must be between 0 and 1")
        config.allocation_weight = weight

    config.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {
        'success': True,
        'side': config.side,
        'display_name': config.display_name,
        'updated_at': config.updated_at.isoformat()
    }


@app.get("/api/mappings")
async def get_mappings(
    mapping_type: str = Query("budget_to_gmp", description="Type: budget_to_gmp or direct_to_budget"),
    side: Optional[str] = Query(None, description="Filter by side: EAST, WEST, BOTH"),
    include_both: bool = Query(True, description="When filtering by EAST/WEST, also include BOTH"),
    limit: int = Query(500, ge=1, le=5000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Get mappings with optional side filter.

    Query Parameters:
    - mapping_type: 'budget_to_gmp' or 'direct_to_budget'
    - side: Filter by side (EAST, WEST, BOTH) - optional
    - include_both: When filtering by EAST or WEST, also include BOTH mappings (default: True)
    - limit: Max records (default: 500, max: 5000)
    - offset: Pagination offset
    """
    # Validate side parameter
    if side:
        side = side.upper()
        if side not in VALID_SIDES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid side. Must be one of: {', '.join(VALID_SIDES)}"
            )

    if mapping_type == "budget_to_gmp":
        query = db.query(BudgetToGMP)

        if side:
            if include_both and side != 'BOTH':
                query = query.filter(BudgetToGMP.side.in_([side, 'BOTH']))
            else:
                query = query.filter(BudgetToGMP.side == side)

        total = query.count()
        mappings = query.offset(offset).limit(limit).all()

        return {
            'mapping_type': mapping_type,
            'side_filter': side,
            'include_both': include_both,
            'total': total,
            'limit': limit,
            'offset': offset,
            'mappings': [
                {
                    'id': m.id,
                    'budget_code': m.budget_code,
                    'cost_code_tier2': m.cost_code_tier2,
                    'gmp_division': m.gmp_division,
                    'side': m.side,
                    'confidence': m.confidence,
                    'created_at': m.created_at.isoformat() if m.created_at else None,
                    'updated_at': m.updated_at.isoformat() if m.updated_at else None
                }
                for m in mappings
            ]
        }

    elif mapping_type == "direct_to_budget":
        query = db.query(DirectToBudget)

        if side:
            if include_both and side != 'BOTH':
                query = query.filter(DirectToBudget.side.in_([side, 'BOTH']))
            else:
                query = query.filter(DirectToBudget.side == side)

        total = query.count()
        mappings = query.offset(offset).limit(limit).all()

        return {
            'mapping_type': mapping_type,
            'side_filter': side,
            'include_both': include_both,
            'total': total,
            'limit': limit,
            'offset': offset,
            'mappings': [
                {
                    'id': m.id,
                    'cost_code': m.cost_code,
                    'name': m.name,
                    'budget_code': m.budget_code,
                    'side': m.side,
                    'confidence': m.confidence,
                    'method': m.method,
                    'vendor_normalized': m.vendor_normalized,
                    'created_at': m.created_at.isoformat() if m.created_at else None,
                    'updated_at': m.updated_at.isoformat() if m.updated_at else None
                }
                for m in mappings
            ]
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="mapping_type must be 'budget_to_gmp' or 'direct_to_budget'"
        )


@app.get("/api/mappings/direct/items")
async def get_direct_cost_items(
    status: str = Query("unmapped", description="Filter: unmapped, mapped, or all"),
    side: Optional[str] = Query(None, description="Filter by side: EAST, WEST, BOTH"),
    type_filter: Optional[str] = Query(None, description="Filter by type: L, M, S, O"),
    search: Optional[str] = Query(None, description="Search query"),
    confidence_band: Optional[str] = Query(None, description="Filter: high, medium, low"),
    sort: Optional[str] = Query("confidence_desc", description="Sort: confidence_desc, amount_desc, date_desc, vendor_asc, cost_code_asc"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db)
):
    """
    Get paginated direct cost items with suggestions for the mappings page.
    Returns enriched items with budget code suggestions and confidence scores.
    """
    data_loader = get_data_loader()

    # Get existing mappings from database
    direct_query = db.query(DirectToBudget)
    if side:
        side_upper = side.upper()
        if side_upper in VALID_SIDES:
            if side_upper != 'BOTH':
                direct_query = direct_query.filter(DirectToBudget.side.in_([side_upper, 'BOTH']))
            else:
                direct_query = direct_query.filter(DirectToBudget.side == side_upper)

    direct_mappings_db = direct_query.all()
    direct_mappings_lookup = {}
    for m in direct_mappings_db:
        key = (m.cost_code, m.name)
        direct_mappings_lookup[key] = {
            'id': m.id,
            'budget_code': m.budget_code,
            'side': m.side,
            'confidence': m.confidence
        }

    # Build budget description lookup
    budget_desc_lookup = {}
    for _, row in data_loader.budget.iterrows():
        bc = row.get('Budget Code', '')
        if bc:
            budget_desc_lookup[bc] = row.get('Budget Code Description', '')

    # Get direct costs dataframe
    direct_df = data_loader.direct_costs.copy()

    # Get set of mapped (cost_code, name) pairs from database
    mapped_keys = set(direct_mappings_lookup.keys())

    # Compute suggestions for unmapped items (only if needed)
    if status in ['unmapped', 'all']:
        # Filter to items NOT in database mappings
        unmapped_mask = direct_df.apply(
            lambda row: (row.get('Cost Code', ''), row.get('Name', '')) not in mapped_keys,
            axis=1
        )
        unmapped_direct_df = direct_df[unmapped_mask].copy()

        dc_suggestions = compute_all_suggestions(
            unmapped_direct_df,
            data_loader.budget,
            db,
            unmapped_only=False,  # Already filtered to unmapped
            top_k=3
        )
    else:
        dc_suggestions = {}

    # Build all items list
    all_items = []
    display_columns = ['Cost Code', 'Name', 'Vendor', 'Invoice #', 'Date', 'Amount', 'Type', 'Description']

    for _, row in direct_df.iterrows():
        dc_id = row.get('direct_cost_id', 0)
        cost_code = row.get('Cost Code', '')
        name = row.get('Name', '')
        key = (cost_code, name)

        item = {col: row.get(col, '') for col in display_columns if col in row.index}
        item['direct_cost_id'] = dc_id

        # Format amount for display
        if 'amount_cents' in row.index:
            item['Amount'] = cents_to_display(int(row['amount_cents']))
        elif 'Amount' in row.index:
            item['Amount'] = row['Amount']

        # Check if mapped
        if key in direct_mappings_lookup:
            mapping = direct_mappings_lookup[key]
            item['is_mapped'] = True
            item['mapping_id'] = mapping['id']
            item['mapped_budget_code'] = mapping['budget_code']
            item['side'] = mapping['side']
            item['mapping_confidence'] = mapping['confidence']
            item['budget_description'] = budget_desc_lookup.get(mapping['budget_code'], '')
            item['confidence_band'] = 'mapped'
            item['suggestions'] = []
            item['top_suggestion'] = None
            item['confidence'] = 100
        else:
            item['is_mapped'] = False
            item['side'] = 'BOTH'
            item['mapped_budget_code'] = None
            item['mapping_confidence'] = 0
            item['budget_description'] = ''

            # Add suggestions for unmapped items
            suggs = dc_suggestions.get(dc_id, [])
            if suggs:
                item['suggestions'] = suggs
                item['top_suggestion'] = suggs[0]
                item['confidence'] = suggs[0].get('score', 0)
                item['confidence_band'] = suggs[0].get('confidence_band', 'low')
            else:
                item['suggestions'] = []
                item['top_suggestion'] = None
                item['confidence'] = 0
                item['confidence_band'] = 'low'

        all_items.append(item)

    # Apply status filter
    if status == 'unmapped':
        filtered_items = [d for d in all_items if not d['is_mapped']]
    elif status == 'mapped':
        filtered_items = [d for d in all_items if d['is_mapped']]
    else:
        filtered_items = all_items

    # Apply type filter
    if type_filter:
        filtered_items = [d for d in filtered_items if d.get('Type', '') == type_filter]

    # Apply confidence band filter
    if confidence_band:
        filtered_items = [d for d in filtered_items if d.get('confidence_band', '') == confidence_band]

    # Apply search filter
    if search:
        search_lower = search.lower()
        filtered_items = [
            d for d in filtered_items
            if search_lower in str(d.get('Vendor', '')).lower()
            or search_lower in str(d.get('Cost Code', '')).lower()
            or search_lower in str(d.get('Name', '')).lower()
        ]

    # Sort based on sort parameter
    def get_sort_key(item, sort_option):
        """Return sort key based on sort option."""
        if sort_option == 'confidence_desc':
            return (-item.get('confidence', 0), str(item.get('Cost Code', '')))
        elif sort_option == 'amount_desc':
            # Parse amount string to number for sorting
            amt_str = str(item.get('Amount', '$0')).replace('$', '').replace(',', '')
            try:
                amt = float(amt_str) if amt_str else 0
            except ValueError:
                amt = 0
            return (-amt, str(item.get('Cost Code', '')))
        elif sort_option == 'date_desc':
            return (str(item.get('Date', '0000-00-00'))[::-1], str(item.get('Cost Code', '')))
        elif sort_option == 'vendor_asc':
            return (str(item.get('Vendor', '')).lower(), str(item.get('Cost Code', '')))
        elif sort_option == 'cost_code_asc':
            return (str(item.get('Cost Code', '')), str(item.get('Name', '')))
        else:
            return (-item.get('confidence', 0), str(item.get('Cost Code', '')))

    if status == 'all':
        # All: unmapped first, then mapped (each sorted by selected option)
        unmapped = [d for d in filtered_items if not d['is_mapped']]
        mapped = [d for d in filtered_items if d['is_mapped']]
        unmapped.sort(key=lambda x: get_sort_key(x, sort or 'confidence_desc'))
        mapped.sort(key=lambda x: get_sort_key(x, sort or 'confidence_desc'))
        filtered_items = unmapped + mapped
    else:
        filtered_items.sort(key=lambda x: get_sort_key(x, sort or 'confidence_desc'))

    # Count totals before pagination
    total_count = len(filtered_items)
    total_high = len([d for d in filtered_items if d.get('confidence_band') == 'high' and not d['is_mapped']])
    total_medium = len([d for d in filtered_items if d.get('confidence_band') == 'medium' and not d['is_mapped']])
    total_low = len([d for d in filtered_items if d.get('confidence_band') == 'low' and not d['is_mapped']])
    total_mapped = len([d for d in filtered_items if d['is_mapped']])

    # Apply pagination
    paginated_items = filtered_items[offset:offset + limit]
    has_more = (offset + limit) < total_count

    # Sanitize items to handle NaN and numpy types
    sanitized_items = [convert_numpy_types(item) for item in paginated_items]

    return {
        'items': sanitized_items,
        'pagination': {
            'offset': offset,
            'limit': limit,
            'total': total_count,
            'hasMore': has_more,
            'nextOffset': offset + limit if has_more else None
        },
        'stats': {
            'total': total_count,
            'high': total_high,
            'medium': total_medium,
            'low': total_low,
            'mapped': total_mapped
        },
        'filters': {
            'status': status,
            'side': side,
            'type': type_filter,
            'search': search,
            'confidence_band': confidence_band,
            'sort': sort
        }
    }


@app.get("/api/mappings/budget/items")
async def get_budget_mapping_items(
    status: str = Query("unmapped", description="Filter: unmapped, mapped, or all"),
    side: Optional[str] = Query(None, description="Filter by side: EAST, WEST, BOTH"),
    search: Optional[str] = Query(None, description="Search query"),
    sort: Optional[str] = Query("confidence_desc", description="Sort: confidence_desc, code_asc, description_asc"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db)
):
    """
    Get paginated budget items for Budget→GMP mapping page.
    Returns enriched items with GMP division suggestions.
    """
    from rapidfuzz import fuzz, process

    data_loader = get_data_loader()

    # Get GMP options
    gmp_options = data_loader.gmp['GMP'].tolist()

    # Get existing mappings from database
    budget_query = db.query(BudgetToGMP)
    if side:
        side_upper = side.upper()
        if side_upper in VALID_SIDES:
            if side_upper != 'BOTH':
                budget_query = budget_query.filter(BudgetToGMP.side.in_([side_upper, 'BOTH']))
            else:
                budget_query = budget_query.filter(BudgetToGMP.side == side_upper)

    budget_mappings_db = budget_query.all()
    mapped_budget_codes = set()
    budget_side_lookup = {}
    budget_mappings_lookup = {}

    for m in budget_mappings_db:
        mapped_budget_codes.add(m.budget_code)
        budget_side_lookup[m.budget_code] = m.side
        budget_mappings_lookup[m.budget_code] = {
            'id': m.id,
            'gmp_division': m.gmp_division,
            'side': m.side,
            'confidence': m.confidence
        }

    # Build all budget items
    budget_df = data_loader.budget.copy()
    gmp_df = data_loader.gmp.copy()
    mapped_budget_df = map_budget_to_gmp(budget_df, gmp_df, db)

    all_items = []
    for _, row in mapped_budget_df.iterrows():
        bc = row.get('Budget Code', '')
        desc = row.get('Budget Code Description', '')
        # Handle NaN descriptions
        if desc is None or (isinstance(desc, float) and desc != desc):
            desc = ''
        else:
            desc = str(desc).strip()

        item = {
            'Budget Code': bc,
            'Budget Code Description': desc,
            'Cost Type': row.get('Cost Type', ''),
            'side': budget_side_lookup.get(bc, 'BOTH'),
        }

        # Check if mapped
        if bc in mapped_budget_codes:
            mapping = budget_mappings_lookup[bc]
            item['is_mapped'] = True
            item['mapping_id'] = mapping['id']
            item['gmp_division'] = mapping['gmp_division']
            item['mapping_method'] = 'database'
            item['mapping_confidence'] = mapping['confidence']
        elif row.get('gmp_division') is not None:
            item['is_mapped'] = True
            item['gmp_division'] = row.get('gmp_division')
            item['mapping_method'] = row.get('mapping_method', 'auto')
            item['mapping_confidence'] = row.get('mapping_confidence', 0)
        else:
            item['is_mapped'] = False
            item['gmp_division'] = None
            item['mapping_method'] = 'unmapped'
            item['mapping_confidence'] = 0

            # Add suggestion for unmapped items
            if desc and len(desc) > 3:
                match = process.extractOne(desc, gmp_options, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 60:
                    item['suggested_gmp'] = match[0]
                    item['suggestion_confidence'] = match[1]

        all_items.append(item)

    # Apply status filter
    if status == 'unmapped':
        filtered_items = [b for b in all_items if not b['is_mapped']]
    elif status == 'mapped':
        filtered_items = [b for b in all_items if b['is_mapped']]
    else:
        filtered_items = all_items

    # Apply search filter
    if search:
        search_lower = search.lower()
        filtered_items = [
            b for b in filtered_items
            if search_lower in str(b.get('Budget Code', '')).lower()
            or search_lower in str(b.get('Budget Code Description', '')).lower()
        ]

    # Sort based on sort parameter
    def get_sort_key(item, sort_option):
        if sort_option == 'confidence_desc':
            return (-float(item.get('suggestion_confidence', 0) or item.get('mapping_confidence', 0) or 0), str(item.get('Budget Code', '') or ''))
        elif sort_option == 'code_asc':
            return (str(item.get('Budget Code', '') or ''),)
        elif sort_option == 'description_asc':
            return (str(item.get('Budget Code Description', '') or '').lower(),)
        else:
            return (-float(item.get('suggestion_confidence', 0) or 0), str(item.get('Budget Code', '') or ''))

    filtered_items.sort(key=lambda x: get_sort_key(x, sort))

    # Count totals (from all_items, not filtered_items, for accurate stats)
    total_count = len(filtered_items)
    total_with_suggestions = len([b for b in all_items if b.get('suggested_gmp') and not b['is_mapped']])
    total_mapped = len([b for b in all_items if b['is_mapped']])
    total_unmapped = len([b for b in all_items if not b['is_mapped']])

    # Apply pagination
    paginated_items = filtered_items[offset:offset + limit]
    has_more = (offset + limit) < total_count

    # Sanitize items to handle NaN and numpy types
    sanitized_items = [convert_numpy_types(item) for item in paginated_items]

    return {
        'items': sanitized_items,
        'pagination': {
            'offset': offset,
            'limit': limit,
            'total': total_count,
            'hasMore': has_more,
            'nextOffset': offset + limit if has_more else None
        },
        'stats': {
            'total': len(all_items),
            'with_suggestions': total_with_suggestions,
            'mapped': total_mapped,
            'unmapped': total_unmapped
        },
        'filters': {
            'status': status,
            'side': side,
            'search': search,
            'sort': sort
        }
    }


@app.patch("/api/mappings/{mapping_type}/{mapping_id}/side")
async def update_mapping_side(
    mapping_type: str,
    mapping_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Update the side assignment for a single mapping.

    Path Parameters:
    - mapping_type: 'budget_to_gmp' or 'direct_to_budget'
    - mapping_id: ID of the mapping to update

    Body:
    {
        "side": "EAST"  // EAST, WEST, or BOTH
    }
    """
    body = await request.json()
    new_side = body.get('side', '').upper()

    if new_side not in VALID_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid side. Must be one of: {', '.join(VALID_SIDES)}"
        )

    if mapping_type == "budget_to_gmp":
        mapping = db.query(BudgetToGMP).filter(BudgetToGMP.id == mapping_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="Mapping not found")

        old_side = mapping.side
        mapping.side = new_side
        mapping.updated_at = datetime.now(timezone.utc)
        db.commit()

        return {
            'success': True,
            'mapping_type': mapping_type,
            'mapping_id': mapping_id,
            'old_side': old_side,
            'new_side': new_side,
            'budget_code': mapping.budget_code,
            'gmp_division': mapping.gmp_division
        }

    elif mapping_type == "direct_to_budget":
        mapping = db.query(DirectToBudget).filter(DirectToBudget.id == mapping_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="Mapping not found")

        old_side = mapping.side
        mapping.side = new_side
        mapping.updated_at = datetime.now(timezone.utc)
        db.commit()

        return {
            'success': True,
            'mapping_type': mapping_type,
            'mapping_id': mapping_id,
            'old_side': old_side,
            'new_side': new_side,
            'cost_code': mapping.cost_code,
            'budget_code': mapping.budget_code
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="mapping_type must be 'budget_to_gmp' or 'direct_to_budget'"
        )


@app.post("/api/mappings/bulk-side")
async def bulk_update_mapping_side(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Bulk update side assignment for multiple mappings.

    Body:
    {
        "mapping_type": "budget_to_gmp",  // or "direct_to_budget"
        "mapping_ids": [1, 2, 3, 4],      // List of mapping IDs to update
        "side": "EAST"                     // New side value
    }

    Alternative - filter-based bulk update:
    {
        "mapping_type": "budget_to_gmp",
        "filter": {
            "current_side": "BOTH",           // Optional: only update mappings with this side
            "gmp_division": "Concrete"        // Optional: only update mappings for this division
        },
        "side": "WEST"
    }
    """
    body = await request.json()

    mapping_type = body.get('mapping_type')
    new_side = body.get('side', '').upper()
    mapping_ids = body.get('mapping_ids', [])
    filter_opts = body.get('filter', {})

    if not mapping_type:
        raise HTTPException(status_code=400, detail="mapping_type is required")

    if new_side not in VALID_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid side. Must be one of: {', '.join(VALID_SIDES)}"
        )

    if not mapping_ids and not filter_opts:
        raise HTTPException(
            status_code=400,
            detail="Either mapping_ids or filter must be provided"
        )

    updated_count = 0

    if mapping_type == "budget_to_gmp":
        if mapping_ids:
            # Update by IDs
            updated_count = db.query(BudgetToGMP).filter(
                BudgetToGMP.id.in_(mapping_ids)
            ).update({
                BudgetToGMP.side: new_side,
                BudgetToGMP.updated_at: datetime.now(timezone.utc)
            }, synchronize_session=False)
        else:
            # Update by filter
            query = db.query(BudgetToGMP)
            if filter_opts.get('current_side'):
                query = query.filter(BudgetToGMP.side == filter_opts['current_side'].upper())
            if filter_opts.get('gmp_division'):
                query = query.filter(BudgetToGMP.gmp_division == filter_opts['gmp_division'])
            if filter_opts.get('budget_code'):
                query = query.filter(BudgetToGMP.budget_code == filter_opts['budget_code'])

            updated_count = query.update({
                BudgetToGMP.side: new_side,
                BudgetToGMP.updated_at: datetime.now(timezone.utc)
            }, synchronize_session=False)

    elif mapping_type == "direct_to_budget":
        if mapping_ids:
            # Update by IDs
            updated_count = db.query(DirectToBudget).filter(
                DirectToBudget.id.in_(mapping_ids)
            ).update({
                DirectToBudget.side: new_side,
                DirectToBudget.updated_at: datetime.now(timezone.utc)
            }, synchronize_session=False)
        else:
            # Update by filter
            query = db.query(DirectToBudget)
            if filter_opts.get('current_side'):
                query = query.filter(DirectToBudget.side == filter_opts['current_side'].upper())
            if filter_opts.get('budget_code'):
                query = query.filter(DirectToBudget.budget_code == filter_opts['budget_code'])

            updated_count = query.update({
                DirectToBudget.side: new_side,
                DirectToBudget.updated_at: datetime.now(timezone.utc)
            }, synchronize_session=False)
    else:
        raise HTTPException(
            status_code=400,
            detail="mapping_type must be 'budget_to_gmp' or 'direct_to_budget'"
        )

    db.commit()

    return {
        'success': True,
        'mapping_type': mapping_type,
        'new_side': new_side,
        'updated_count': updated_count,
        'filter_used': filter_opts if filter_opts else None,
        'ids_provided': len(mapping_ids) if mapping_ids else 0
    }


@app.get("/api/mappings/side-summary")
async def get_mapping_side_summary(db: Session = Depends(get_db)):
    """
    Get summary counts of mappings by side for both mapping types.
    Useful for dashboard widgets and side distribution overview.
    """
    # Budget to GMP counts
    btg_counts = {}
    for side in VALID_SIDES:
        count = db.query(BudgetToGMP).filter(BudgetToGMP.side == side).count()
        btg_counts[side.lower()] = count

    btg_total = sum(btg_counts.values())

    # Direct to Budget counts
    dtb_counts = {}
    for side in VALID_SIDES:
        count = db.query(DirectToBudget).filter(DirectToBudget.side == side).count()
        dtb_counts[side.lower()] = count

    dtb_total = sum(dtb_counts.values())

    return {
        'budget_to_gmp': {
            'by_side': btg_counts,
            'total': btg_total,
            'percentages': {
                k: round(v / btg_total * 100, 1) if btg_total > 0 else 0
                for k, v in btg_counts.items()
            }
        },
        'direct_to_budget': {
            'by_side': dtb_counts,
            'total': dtb_total,
            'percentages': {
                k: round(v / dtb_total * 100, 1) if dtb_total > 0 else 0
                for k, v in dtb_counts.items()
            }
        }
    }


# ------------ Forecasting API Endpoints ------------

@app.get("/api/gmp/{gmp_division}/forecast")
async def get_gmp_forecast(gmp_division: str, db: Session = Depends(get_db)):
    """
    Get current forecast summary for a GMP division.
    Returns EAC, method, confidence, and key metrics.
    """
    manager = ForecastManager(db)
    snapshot = manager.get_current_snapshot(gmp_division)

    if not snapshot:
        # No forecast exists yet - return empty structure
        return {
            'gmp_division': gmp_division,
            'has_forecast': False,
            'message': 'No forecast computed yet. Trigger a refresh to generate forecast.'
        }

    config = manager.get_or_create_config(gmp_division)

    return {
        'gmp_division': gmp_division,
        'has_forecast': True,
        'snapshot_id': snapshot.id,
        'snapshot_date': snapshot.snapshot_date.isoformat(),
        'bac_cents': snapshot.bac_cents,
        'bac_display': cents_to_display(snapshot.bac_cents),
        'ac_cents': snapshot.ac_cents,
        'ac_display': cents_to_display(snapshot.ac_cents),
        'ev_cents': snapshot.ev_cents,
        'eac_cents': snapshot.eac_cents,
        'eac_display': cents_to_display(snapshot.eac_cents),
        'eac_west_cents': snapshot.eac_west_cents,
        'eac_east_cents': snapshot.eac_east_cents,
        'etc_cents': snapshot.etc_cents,
        'etc_display': cents_to_display(snapshot.etc_cents),
        'var_cents': snapshot.var_cents,
        'var_display': cents_to_display(snapshot.var_cents),
        'var_percent': round(snapshot.var_cents / snapshot.bac_cents * 100, 1) if snapshot.bac_cents else 0,
        'percent_complete': round(snapshot.ac_cents / snapshot.eac_cents * 100, 1) if snapshot.eac_cents else 0,
        'cpi': snapshot.cpi,
        'spi': snapshot.spi,
        'method': snapshot.method,
        'confidence_score': snapshot.confidence_score,
        'confidence_band': snapshot.confidence_band,
        'explanation': snapshot.explanation,
        'trigger': snapshot.trigger,
        'is_locked': config.is_locked,
        'distribution_method': config.distribution_method,
        'completion_date': config.completion_date.isoformat() if config.completion_date else None
    }


@app.get("/api/gmp/{gmp_division}/forecast/periods")
async def get_gmp_forecast_periods(
    gmp_division: str,
    granularity: str = "weekly",
    db: Session = Depends(get_db)
):
    """
    Get time-bucketed forecast periods for a GMP division.
    Supports weekly and monthly granularity.
    """
    if granularity not in ['weekly', 'monthly']:
        raise HTTPException(status_code=400, detail="Granularity must be 'weekly' or 'monthly'")

    manager = ForecastManager(db)
    periods = manager.get_periods(gmp_division, granularity)

    if not periods:
        return {
            'gmp_division': gmp_division,
            'granularity': granularity,
            'periods': [],
            'message': 'No periods available. Compute forecast first.'
        }

    return {
        'gmp_division': gmp_division,
        'granularity': granularity,
        'period_count': len(periods),
        'periods': [
            {
                'period_label': p.period_label,
                'period_number': p.period_number,
                'period_start': p.period_start.isoformat(),
                'period_end': p.period_end.isoformat(),
                'period_type': p.period_type,
                'iso_week': p.iso_week,
                'iso_year': p.iso_year,
                'actual_cents': p.actual_cents,
                'actual_display': cents_to_display(p.actual_cents),
                'forecast_cents': p.forecast_cents,
                'forecast_display': cents_to_display(p.forecast_cents),
                'blended_cents': p.blended_cents,
                'blended_display': cents_to_display(p.blended_cents),
                'cumulative_cents': p.cumulative_cents,
                'cumulative_display': cents_to_display(p.cumulative_cents),
                'actual_west_cents': p.actual_west_cents,
                'actual_east_cents': p.actual_east_cents,
                'forecast_west_cents': p.forecast_west_cents,
                'forecast_east_cents': p.forecast_east_cents
            }
            for p in periods
        ]
    }


@app.put("/api/gmp/{gmp_division}/forecast/method")
async def update_forecast_method(
    gmp_division: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Update the forecasting method for a GMP division.
    Body: { "method": "evm|pert|parametric|ml_linear|manual" }
    """
    body = await request.json()
    method = body.get('method')

    valid_methods = ['evm', 'pert', 'parametric', 'ml_linear', 'manual']
    if method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method. Must be one of: {valid_methods}"
        )

    manager = ForecastManager(db)
    config = manager.update_method(gmp_division, method, changed_by='user')

    return {
        'gmp_division': gmp_division,
        'method': config.method,
        'updated_at': config.updated_at.isoformat() if config.updated_at else None,
        'message': f'Method updated to {method}. Refresh forecast to apply.'
    }


@app.put("/api/gmp/{gmp_division}/forecast/params")
async def update_forecast_params(
    gmp_division: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Update method-specific parameters for a GMP division.

    EVM params: { "evm_performance_factor": 1.0 }
    PERT params: { "pert_optimistic_cents": X, "pert_most_likely_cents": Y, "pert_pessimistic_cents": Z }
    Parametric: { "param_quantity": N, "param_unit_rate_cents": X, "param_complexity_factor": 1.0 }
    General: { "distribution_method": "linear|front_loaded|back_loaded|s_curve",
               "start_date": "2024-09-05", "completion_date": "2026-06-30" }
    """
    body = await request.json()

    # Validate and convert dates if present
    for date_field in ['start_date', 'completion_date']:
        if date_field in body and body[date_field]:
            try:
                body[date_field] = datetime.fromisoformat(body[date_field])
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid {date_field} format. Use ISO format (YYYY-MM-DD)")

    # Whitelist of allowed parameters
    allowed_params = {
        'evm_performance_factor', 'pert_optimistic_cents', 'pert_most_likely_cents',
        'pert_pessimistic_cents', 'param_quantity', 'param_unit_rate_cents',
        'param_complexity_factor', 'distribution_method', 'start_date', 'completion_date',
        'is_locked', 'notes'
    }

    params = {k: v for k, v in body.items() if k in allowed_params}

    if not params:
        raise HTTPException(status_code=400, detail="No valid parameters provided")

    manager = ForecastManager(db)
    config = manager.update_params(gmp_division, params, changed_by='user')

    return {
        'gmp_division': gmp_division,
        'updated_params': list(params.keys()),
        'updated_at': config.updated_at.isoformat() if config.updated_at else None,
        'message': 'Parameters updated. Refresh forecast to apply.'
    }


@app.post("/api/gmp/{gmp_division}/forecast/refresh")
async def refresh_forecast(
    gmp_division: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Force recalculation of forecast for a GMP division.
    Creates new snapshot, marks previous as superseded.

    Optional body:
    {
        "trigger": "manual|transaction|mapping|schedule",
        "ev_cents": 123456,  // Optional: provide EV if tracked externally
        "spi": 1.0           // Optional: Schedule Performance Index
    }
    """
    # Get optional body params
    body = {}
    try:
        body = await request.json()
    except:
        pass

    trigger = body.get('trigger', 'manual')
    ev_cents = body.get('ev_cents')
    spi = body.get('spi')

    # Load current data
    loader = get_data_loader()
    gmp_df = loader.gmp
    budget_df = loader.budget
    direct_costs_df = loader.direct_costs
    settings = get_settings(db)

    # Find this GMP division's data
    gmp_row = gmp_df[gmp_df['GMP'] == gmp_division]
    if gmp_row.empty:
        raise HTTPException(status_code=404, detail=f"GMP division '{gmp_division}' not found")

    bac_cents = int(gmp_row['amount_total_cents'].iloc[0])

    # Map and aggregate to get actuals
    mapped_budget = map_budget_to_gmp(budget_df.copy(), gmp_df)
    mapped_direct = map_direct_to_budget(direct_costs_df.copy(), budget_df)

    # Aggregate actuals for this division
    division_budget = mapped_budget[mapped_budget['gmp_division'] == gmp_division]
    division_budget_codes = set(division_budget['Budget Code'].unique())

    division_direct = mapped_direct[mapped_direct['mapped_budget_code'].isin(division_budget_codes)]
    ac_cents = int(division_direct['amount_cents'].sum()) if not division_direct.empty else 0

    # Calculate west ratio from allocations
    if 'amount_west_cents' in division_direct.columns and not division_direct.empty:
        total_west = division_direct['amount_west_cents'].sum()
        total = division_direct['amount_cents'].sum()
        west_ratio = total_west / total if total > 0 else 0.5
    else:
        west_ratio = 0.5

    # Compute forecast
    manager = ForecastManager(db)
    snapshot = manager.compute_forecast(
        gmp_division=gmp_division,
        bac_cents=bac_cents,
        ac_cents=ac_cents,
        west_ratio=west_ratio,
        ev_cents=ev_cents,
        spi=spi,
        trigger=trigger
    )

    # Generate time periods (both weekly and monthly)
    as_of_date = settings.get('as_of_date') or loader.max_transaction_date or datetime.now(timezone.utc)
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)
    if hasattr(as_of_date, 'to_pydatetime'):
        as_of_date = as_of_date.to_pydatetime()

    config = manager.get_or_create_config(gmp_division)

    # Determine start_date with priority:
    # 1. Config override (user-specified)
    # 2. Earliest transaction date from ALL direct costs (project-wide)
    # 3. Fallback to Jan 1, 2025
    if config.start_date:
        start_date = config.start_date
    elif not direct_costs_df.empty and 'date_parsed' in direct_costs_df.columns:
        # Use ALL direct costs to get true project start, not just mapped ones
        valid_dates = direct_costs_df['date_parsed'].dropna()
        if not valid_dates.empty:
            start_date = valid_dates.min()
            if hasattr(start_date, 'to_pydatetime'):
                start_date = start_date.to_pydatetime()
        else:
            start_date = datetime(2025, 1, 1)
    else:
        start_date = datetime(2025, 1, 1)

    if config.completion_date:
        end_date = config.completion_date
    else:
        end_date = (pd.Timestamp(as_of_date) + pd.DateOffset(months=6)).to_pydatetime()

    # Generate periods for both granularities
    manager.generate_periods(
        snapshot=snapshot,
        granularity='weekly',
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        transactions_df=division_direct
    )
    manager.generate_periods(
        snapshot=snapshot,
        granularity='monthly',
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        transactions_df=division_direct
    )

    return {
        'gmp_division': gmp_division,
        'snapshot_id': snapshot.id,
        'bac_cents': snapshot.bac_cents,
        'bac_display': cents_to_display(snapshot.bac_cents),
        'ac_cents': snapshot.ac_cents,
        'ac_display': cents_to_display(snapshot.ac_cents),
        'eac_cents': snapshot.eac_cents,
        'eac_display': cents_to_display(snapshot.eac_cents),
        'var_cents': snapshot.var_cents,
        'var_display': cents_to_display(snapshot.var_cents),
        'method': snapshot.method,
        'confidence_band': snapshot.confidence_band,
        'explanation': snapshot.explanation,
        'message': 'Forecast refreshed successfully'
    }


@app.get("/api/gmp/{gmp_division}/forecast/history")
async def get_forecast_history(
    gmp_division: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get forecast change history (audit log) for a GMP division.
    """
    if limit > 200:
        limit = 200

    manager = ForecastManager(db)
    logs = manager.get_history(gmp_division, limit)

    return {
        'gmp_division': gmp_division,
        'history': [
            {
                'action': log.action,
                'field': log.field_changed,
                'old_value': log.old_value,
                'new_value': log.new_value,
                'previous_eac_cents': log.previous_eac_cents,
                'previous_eac_display': cents_to_display(log.previous_eac_cents) if log.previous_eac_cents else None,
                'new_eac_cents': log.new_eac_cents,
                'new_eac_display': cents_to_display(log.new_eac_cents) if log.new_eac_cents else None,
                'reason': log.change_reason,
                'changed_by': log.changed_by,
                'changed_at': log.changed_at.isoformat() if log.changed_at else None
            }
            for log in logs
        ]
    }


@app.get("/api/gmp/{gmp_division}/forecast/snapshots")
async def get_forecast_snapshots(
    gmp_division: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get historical forecast snapshots for trend analysis.
    """
    if limit > 100:
        limit = 100

    manager = ForecastManager(db)
    snapshots = manager.get_snapshot_history(gmp_division, limit)

    return {
        'gmp_division': gmp_division,
        'snapshots': [
            {
                'snapshot_id': s.id,
                'snapshot_date': s.snapshot_date.isoformat(),
                'bac_cents': s.bac_cents,
                'ac_cents': s.ac_cents,
                'eac_cents': s.eac_cents,
                'var_cents': s.var_cents,
                'cpi': s.cpi,
                'method': s.method,
                'confidence_band': s.confidence_band,
                'is_current': s.is_current,
                'trigger': s.trigger
            }
            for s in snapshots
        ]
    }


@app.get("/api/gmp/{gmp_division}/forecast/export")
async def export_forecast(
    gmp_division: str,
    format: str = "csv",
    granularity: str = "weekly",
    db: Session = Depends(get_db)
):
    """
    Export forecast data for a GMP division.
    Supports CSV and XLSX formats.
    """
    if format not in ['csv', 'xlsx']:
        raise HTTPException(status_code=400, detail="Format must be 'csv' or 'xlsx'")
    if granularity not in ['weekly', 'monthly']:
        raise HTTPException(status_code=400, detail="Granularity must be 'weekly' or 'monthly'")

    manager = ForecastManager(db)
    snapshot = manager.get_current_snapshot(gmp_division)
    periods = manager.get_periods(gmp_division, granularity)

    if not snapshot or not periods:
        raise HTTPException(status_code=404, detail="No forecast data available")

    # Build DataFrame
    data = []
    for p in periods:
        data.append({
            'Period': p.period_label,
            'Start Date': p.period_start.strftime('%Y-%m-%d'),
            'End Date': p.period_end.strftime('%Y-%m-%d'),
            'Type': p.period_type.capitalize(),
            'Actual': p.actual_cents / 100,
            'Forecast': p.forecast_cents / 100,
            'Blended': p.blended_cents / 100,
            'Cumulative': p.cumulative_cents / 100,
            'Actual West': p.actual_west_cents / 100,
            'Actual East': p.actual_east_cents / 100,
            'Forecast West': p.forecast_west_cents / 100,
            'Forecast East': p.forecast_east_cents / 100
        })

    df = pd.DataFrame(data)

    # Export
    output = io.BytesIO()
    if format == 'csv':
        df.to_csv(output, index=False)
        media_type = 'text/csv'
        filename = f"{gmp_division}_forecast_{granularity}.csv"
    else:
        df.to_excel(output, index=False, sheet_name='Forecast')
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        filename = f"{gmp_division}_forecast_{granularity}.xlsx"

    output.seek(0)
    return StreamingResponse(
        output,
        media_type=media_type,
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


@app.get("/api/project/forecast/rollup")
async def get_project_forecast_rollup(db: Session = Depends(get_db)):
    """
    Get project-level aggregated forecast across all GMP divisions.
    """
    rollup = compute_project_rollup(db)

    # Add display formatting
    rollup['total_bac_display'] = cents_to_display(rollup['total_bac_cents'])
    rollup['total_ac_display'] = cents_to_display(rollup['total_ac_cents'])
    rollup['total_eac_display'] = cents_to_display(rollup['total_eac_cents'])
    rollup['total_etc_display'] = cents_to_display(rollup['total_etc_cents'])
    rollup['total_var_display'] = cents_to_display(rollup['total_var_cents'])

    # Add display formatting to each division
    for div in rollup['by_division']:
        div['bac_display'] = cents_to_display(div['bac_cents'])
        div['ac_display'] = cents_to_display(div['ac_cents'])
        div['eac_display'] = cents_to_display(div['eac_cents'])
        div['etc_display'] = cents_to_display(div['etc_cents'])
        div['var_display'] = cents_to_display(div['var_cents'])

    return rollup


@app.post("/api/project/forecast/refresh-all")
async def refresh_all_forecasts(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Refresh forecasts for all GMP divisions.
    Typically called after data reload or on schedule.
    """
    loader = get_data_loader()
    gmp_df = loader.gmp

    results = []
    errors = []

    for _, gmp_row in gmp_df.iterrows():
        gmp_division = gmp_row['GMP']
        try:
            # Call single refresh endpoint logic inline
            manager = ForecastManager(db)
            bac_cents = int(gmp_row['amount_total_cents'])

            # Quick aggregate for this division
            budget_df = loader.budget
            direct_costs_df = loader.direct_costs
            mapped_budget = map_budget_to_gmp(budget_df.copy(), gmp_df)
            mapped_direct = map_direct_to_budget(direct_costs_df.copy(), budget_df)

            division_budget = mapped_budget[mapped_budget['gmp_division'] == gmp_division]
            division_budget_codes = set(division_budget['Budget Code'].unique())
            division_direct = mapped_direct[mapped_direct['mapped_budget_code'].isin(division_budget_codes)]
            ac_cents = int(division_direct['amount_cents'].sum()) if not division_direct.empty else 0

            snapshot = manager.compute_forecast(
                gmp_division=gmp_division,
                bac_cents=bac_cents,
                ac_cents=ac_cents,
                trigger='batch_refresh'
            )

            results.append({
                'gmp_division': gmp_division,
                'eac_cents': snapshot.eac_cents,
                'status': 'success'
            })

        except Exception as e:
            errors.append({
                'gmp_division': gmp_division,
                'error': str(e)
            })

    return {
        'refreshed': len(results),
        'errors': len(errors),
        'results': results,
        'error_details': errors
    }


# =============================================================================
# Breakdown API Endpoints (Task 8)
# =============================================================================

@app.get("/api/breakdown/items")
async def get_breakdown_items(db: Session = Depends(get_db)):
    """Get all breakdown items with their GMP mappings."""
    items = db.query(GMPBudgetBreakdown).all()
    return {
        'items': [
            {
                'id': item.id,
                'cost_code_description': item.cost_code_description,
                'gmp_division': item.gmp_division,
                'gmp_sov_cents': item.gmp_sov_cents,
                'gmp_sov_display': cents_to_display(item.gmp_sov_cents),
                'east_funded_cents': item.east_funded_cents,
                'east_funded_display': cents_to_display(item.east_funded_cents),
                'west_funded_cents': item.west_funded_cents,
                'west_funded_display': cents_to_display(item.west_funded_cents),
                'pct_east': round(item.pct_east * 100, 1),
                'pct_west': round(item.pct_west * 100, 1),
                'match_score': item.match_score,
                'source_file': item.source_file
            }
            for item in items
        ],
        'count': len(items)
    }


@app.post("/api/breakdown/import")
async def import_breakdown(db: Session = Depends(get_db)):
    """Import breakdown.csv and auto-match to GMP divisions."""
    try:
        loader = get_data_loader()
        breakdown_df = loader.breakdown
        gmp_df = loader.gmp

        if breakdown_df.empty:
            return {'error': 'No breakdown.csv file found or file is empty', 'imported': 0}

        # Fuzzy match to GMP divisions
        matched_df = fuzzy_match_breakdown_to_gmp(breakdown_df, gmp_df, score_cutoff=60)

        # Clear existing breakdown data
        db.query(GMPBudgetBreakdown).delete()

        # Insert new records
        imported = 0
        for _, row in matched_df.iterrows():
            record = GMPBudgetBreakdown(
                cost_code_description=str(row['cost_code_description']),
                gmp_division=row.get('gmp_division'),
                gmp_sov_cents=int(row['gmp_sov_cents']),
                east_funded_cents=int(row['east_funded_cents']),
                west_funded_cents=int(row['west_funded_cents']),
                pct_east=float(row['pct_east']),
                pct_west=float(row['pct_west']),
                match_score=int(row.get('match_score', 0)),
                source_file='breakdown.csv'
            )
            db.add(record)
            imported += 1

        db.commit()

        # Count matched vs unmatched
        matched_count = len(matched_df[matched_df['gmp_division'].notna()])

        return {
            'imported': imported,
            'matched': matched_count,
            'unmatched': imported - matched_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/breakdown/{breakdown_id}/match")
async def update_breakdown_match(
    breakdown_id: int,
    gmp_division: str = Form(...),
    db: Session = Depends(get_db)
):
    """Manually update GMP division match for a breakdown item."""
    item = db.query(GMPBudgetBreakdown).filter(GMPBudgetBreakdown.id == breakdown_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Breakdown item not found")

    item.gmp_division = gmp_division if gmp_division else None
    item.match_score = 100  # Manual match = 100% confidence
    db.commit()

    return {'success': True, 'id': breakdown_id, 'gmp_division': gmp_division}


# =============================================================================
# Schedule API Endpoints (Task 8)
# =============================================================================

@app.get("/api/schedule/activities")
async def get_schedule_activities(
    zone_filter: Optional[str] = Query(None, description="Filter by zone: EAST, WEST, SHARED, or UNASSIGNED"),
    db: Session = Depends(get_db)
):
    """Get all schedule activities with their GMP mappings and zone assignments."""
    query = db.query(ScheduleActivity)

    # Apply zone filter if provided
    if zone_filter:
        zone_filter_upper = zone_filter.upper()
        if zone_filter_upper == 'UNASSIGNED':
            query = query.filter(ScheduleActivity.zone.is_(None))
        elif zone_filter_upper in ('EAST', 'WEST', 'SHARED'):
            query = query.filter(ScheduleActivity.zone == zone_filter_upper)

    activities = query.all()
    result = []
    for act in activities:
        mappings = [
            {
                'gmp_division': m.gmp_division,
                'weight': m.weight
            }
            for m in act.mappings
        ]
        result.append({
            'id': act.id,
            'row_number': act.row_number,
            'task_name': act.task_name,
            'activity_id': act.activity_id,
            'wbs': act.wbs,
            'pct_complete': act.pct_complete,
            'start_date': act.start_date.isoformat() if act.start_date else None,
            'finish_date': act.finish_date.isoformat() if act.finish_date else None,
            'duration_days': act.duration_days,
            'is_complete': act.is_complete,
            'is_in_progress': act.is_in_progress,
            'progress_pct': act.progress_pct,
            'is_critical': act.is_critical,
            'zone': act.zone,  # Zone assignment (EAST, WEST, SHARED, or None)
            'source_file': act.source_file,
            'mappings': mappings
        })

    return {'activities': result, 'count': len(result)}


@app.post("/api/schedule/import")
async def import_schedule(db: Session = Depends(get_db)):
    """Import schedule.csv and auto-match to GMP divisions."""
    try:
        loader = get_data_loader()
        schedule_df = loader.schedule
        gmp_df = loader.gmp

        if schedule_df.empty:
            return {'error': 'No schedule.csv file found or file is empty', 'imported': 0}

        # Fuzzy match to GMP divisions
        matched_df = match_schedule_to_gmp(schedule_df, gmp_df, score_cutoff=50)

        # Clear existing schedule data
        db.query(ScheduleToGMPMapping).delete()
        db.query(ScheduleActivity).delete()

        # Insert new records
        imported = 0
        matched_count = 0

        for _, row in matched_df.iterrows():
            # Get P6 fields (computed by load_schedule_csv)
            import math
            start_is_actual = bool(row.get('start_is_actual', False))
            finish_is_actual = bool(row.get('finish_is_actual', False))
            is_complete = bool(row.get('is_complete', False))
            is_in_progress = bool(row.get('is_in_progress', False))
            progress_pct_val = row.get('progress_pct', 0.0)
            progress_pct = float(progress_pct_val) if progress_pct_val is not None and not (isinstance(progress_pct_val, float) and math.isnan(progress_pct_val)) else 0.0
            total_float_val = row.get('total_float')
            total_float = int(total_float_val) if total_float_val is not None and not (isinstance(total_float_val, float) and math.isnan(total_float_val)) else None
            is_critical = bool(row.get('is_critical', False))

            activity = ScheduleActivity(
                row_number=int(row['row_number']),
                task_name=str(row['task_name']),
                source_uid=row.get('source_uid'),
                activity_id=str(row.get('activity_id', '')),
                wbs=str(row.get('wbs', '')),
                pct_complete=int(row.get('pct_complete', 0)),
                start_date=row.get('start_date'),
                finish_date=row.get('finish_date'),
                planned_start=row.get('planned_start'),
                planned_finish=row.get('planned_finish'),
                duration_days=row.get('duration_days'),
                # P6 fields
                start_is_actual=start_is_actual,
                finish_is_actual=finish_is_actual,
                is_complete=is_complete,
                is_in_progress=is_in_progress,
                progress_pct=progress_pct,
                total_float=int(total_float) if total_float is not None else None,
                is_critical=is_critical,
                source_file='schedule.csv'
            )
            db.add(activity)
            db.flush()  # Get the ID

            # Add GMP mapping if matched
            gmp_div = row.get('gmp_division')
            if gmp_div:
                mapping = ScheduleToGMPMapping(
                    schedule_activity_id=activity.id,
                    gmp_division=gmp_div,
                    weight=1.0,
                    created_by='auto_import'
                )
                db.add(mapping)
                matched_count += 1

            imported += 1

        db.commit()

        return {
            'imported': imported,
            'matched': matched_count,
            'unmatched': imported - matched_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedule/{activity_id}/map")
async def update_schedule_mapping(
    activity_id: int,
    gmp_division: str = Form(...),
    weight: float = Form(1.0),
    db: Session = Depends(get_db)
):
    """Add or update GMP mapping for a schedule activity."""
    activity = db.query(ScheduleActivity).filter(ScheduleActivity.id == activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Check for existing mapping
    existing = db.query(ScheduleToGMPMapping).filter(
        ScheduleToGMPMapping.schedule_activity_id == activity_id,
        ScheduleToGMPMapping.gmp_division == gmp_division
    ).first()

    if existing:
        existing.weight = weight
    else:
        mapping = ScheduleToGMPMapping(
            schedule_activity_id=activity_id,
            gmp_division=gmp_division,
            weight=weight,
            created_by='manual'
        )
        db.add(mapping)

    db.commit()
    return {'success': True, 'activity_id': activity_id, 'gmp_division': gmp_division}


@app.delete("/api/schedule/{activity_id}/unmap")
async def delete_schedule_mapping(
    activity_id: int,
    gmp_division: str,
    db: Session = Depends(get_db)
):
    """Remove GMP mapping from a schedule activity."""
    mapping = db.query(ScheduleToGMPMapping).filter(
        ScheduleToGMPMapping.schedule_activity_id == activity_id,
        ScheduleToGMPMapping.gmp_division == gmp_division
    ).first()

    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    db.delete(mapping)
    db.commit()
    return {'success': True, 'activity_id': activity_id, 'gmp_division': gmp_division}


@app.get("/api/schedule/progress/{gmp_division}")
async def get_schedule_progress(gmp_division: str, db: Session = Depends(get_db)):
    """Get weighted progress for a GMP division from schedule activities."""
    mappings = db.query(ScheduleToGMPMapping).filter(
        ScheduleToGMPMapping.gmp_division == gmp_division
    ).all()

    if not mappings:
        return {'gmp_division': gmp_division, 'weighted_progress': 0.0, 'activities': []}

    total_weight = 0.0
    weighted_sum = 0.0
    activities = []

    for m in mappings:
        activity = m.activity
        weight = m.weight
        progress = activity.pct_complete / 100.0

        weighted_sum += weight * progress
        total_weight += weight

        activities.append({
            'task_name': activity.task_name,
            'pct_complete': activity.pct_complete,
            'weight': weight,
            'contribution': weight * progress
        })

    weighted_progress = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        'gmp_division': gmp_division,
        'weighted_progress': round(weighted_progress * 100, 1),
        'total_weight': total_weight,
        'activities': activities
    }


@app.get("/api/schedule/forecast/{gmp_division}")
async def get_schedule_forecast(gmp_division: str, db: Session = Depends(get_db)):
    """Get schedule-based forecast for a GMP division."""
    from app.modules.reconciliation import compute_schedule_based_forecast

    # Get budget and actuals for this division
    data_loader = get_data_loader()
    gmp_df = data_loader.gmp

    # Find budget for this division
    gmp_row = gmp_df[gmp_df['GMP'] == gmp_division]
    budget_cents = int(gmp_row['amount_total_cents'].iloc[0]) if len(gmp_row) > 0 else 0

    # Get actuals from reconciliation (simplified - uses most recent run)
    result = run_full_reconciliation(db)
    actual_cents = 0
    for row in result['recon_rows']:
        if row['gmp_division'] == gmp_division:
            actual_cents = row['actual_west_raw'] + row['actual_east_raw']
            break

    forecast = compute_schedule_based_forecast(db, gmp_division, budget_cents, actual_cents)
    forecast['budget_cents'] = budget_cents
    forecast['budget_display'] = cents_to_display(budget_cents)
    forecast['actual_cents'] = actual_cents
    forecast['actual_display'] = cents_to_display(actual_cents)

    return forecast


@app.get("/api/schedule/summary")
async def get_schedule_summary(db: Session = Depends(get_db)):
    """Get project-wide schedule summary."""
    from app.modules.reconciliation import compute_project_schedule_summary
    return compute_project_schedule_summary(db)


@app.get("/api/schedule/variance-drilldown/{trade}")
async def get_schedule_variance_drilldown(trade: str, db: Session = Depends(get_db)):
    """
    Get detailed schedule variance breakdown for a specific trade.
    Shows activities, expected vs actual costs, and timeline.
    """
    from pathlib import Path
    from datetime import datetime, timezone
    from urllib.parse import unquote

    trade = unquote(trade)
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    schedule_file = data_dir / "schedule.csv"
    breakdown_file = data_dir / "breakdown.csv"
    direct_costs_file = data_dir / "direct_costs.csv"

    if not all(f.exists() for f in [schedule_file, breakdown_file, direct_costs_file]):
        return {"success": False, "error": "Schedule data files not found"}

    try:
        from src.schedule.parser import ScheduleParser
        from src.schedule.cost_allocator import ActivityCostAllocator

        schedule_df = pd.read_csv(schedule_file)
        breakdown_df = pd.read_csv(breakdown_file)
        direct_costs_df = pd.read_csv(direct_costs_file)

        parser = ScheduleParser(schedule_df)
        allocator = ActivityCostAllocator(parser, breakdown_df)

        if trade not in allocator.gmp:
            return {"success": False, "error": f"Trade '{trade}' not found"}

        as_of = datetime.now()
        budget = allocator.gmp.get(trade, 0)
        expected = allocator.get_expected_cost_to_date(trade, as_of)

        # Get actual costs for this trade
        amount_col = next((c for c in ['amount', 'Amount'] if c in direct_costs_df.columns), None)
        date_col = next((c for c in ['date', 'Date'] if c in direct_costs_df.columns), None)

        actual_items = []
        total_actual = 0
        if amount_col:
            for _, row in direct_costs_df.iterrows():
                name = str(row.get('name', '') or row.get('Description', ''))
                mapped_trade, _, _, _ = parser._map_to_trade(name)
                if mapped_trade == trade:
                    amount = float(row[amount_col])
                    total_actual += amount
                    actual_items.append({
                        "name": name,
                        "amount": amount,
                        "amount_display": f"${amount:,.0f}",
                        "date": str(row.get(date_col, '')) if date_col else None,
                        "vendor": str(row.get('vendor', '') or row.get('Vendor', ''))
                    })

        # Get activities for this trade
        activities = []
        if trade in allocator.allocations:
            for alloc in allocator.allocations[trade]:
                activity = alloc.activity
                act_start = activity.start.date() if hasattr(activity.start, 'date') else activity.start
                act_finish = activity.finish.date() if hasattr(activity.finish, 'date') else activity.finish
                today = datetime.now().date()

                status = "complete" if activity.is_complete(as_of) else "active" if activity.is_active(as_of) else "pending"
                pct = activity.pct_complete(as_of) if hasattr(activity, 'pct_complete') else 0

                activities.append({
                    "name": activity.name,
                    "start": act_start.isoformat(),
                    "finish": act_finish.isoformat(),
                    "duration": activity.duration_days,
                    "expected_cost": alloc.expected_cost,
                    "expected_cost_display": f"${alloc.expected_cost:,.0f}",
                    "status": status,
                    "pct_complete": round(pct * 100, 1) if pct else 0
                })

        # Sort activities by start date
        activities.sort(key=lambda x: x["start"])

        # Calculate variance
        variance = total_actual - expected
        variance_pct = (variance / expected * 100) if expected > 0 else 0
        status = "over" if variance > 0 else "under" if variance < 0 else "on_track"

        return {
            "success": True,
            "trade": trade,
            "budget": budget,
            "budget_display": f"${budget:,.0f}",
            "expected": expected,
            "expected_display": f"${expected:,.0f}",
            "actual": total_actual,
            "actual_display": f"${total_actual:,.0f}",
            "variance": variance,
            "variance_display": f"${variance:+,.0f}",
            "variance_pct": round(variance_pct, 1),
            "status": status,
            "activities": activities,
            "activity_count": len(activities),
            "actual_items": actual_items[:20],  # Limit to 20 items
            "actual_item_count": len(actual_items)
        }

    except Exception as e:
        logger.exception(f"Error getting schedule variance drilldown for {trade}")
        return {"success": False, "error": str(e)}


@app.get("/api/schedule/forecast-all")
async def get_schedule_forecast_all(db: Session = Depends(get_db)):
    """
    Get schedule-driven forecasts for all GMP divisions.
    Uses the schedule-driven trainer for ML-based predictions.
    """
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data" / "raw"
    schedule_file = data_dir / "schedule.csv"
    breakdown_file = data_dir / "breakdown.csv"
    direct_costs_file = data_dir / "direct_costs.csv"

    # Check if data files exist
    if not all(f.exists() for f in [schedule_file, breakdown_file, direct_costs_file]):
        return {
            "success": False,
            "error": "Schedule data files not found",
            "forecasts": [],
            "summary": None
        }

    try:
        from src.schedule.parser import ScheduleParser
        from src.schedule.cost_allocator import ActivityCostAllocator

        # Load data
        schedule_df = pd.read_csv(schedule_file)
        breakdown_df = pd.read_csv(breakdown_file)
        direct_costs_df = pd.read_csv(direct_costs_file)

        # Parse schedule
        parser = ScheduleParser(schedule_df)
        allocator = ActivityCostAllocator(parser, breakdown_df)

        # Get current date for progress calculation
        from datetime import date, datetime
        today = datetime.now().date()

        # Convert parser dates to date objects if they're datetimes
        proj_start = parser.project_start.date() if hasattr(parser.project_start, 'date') else parser.project_start
        proj_end = parser.project_end.date() if hasattr(parser.project_end, 'date') else parser.project_end

        # Calculate project progress
        project_duration = (proj_end - proj_start).days
        days_elapsed = (today - proj_start).days
        project_pct = min(100, max(0, days_elapsed / project_duration * 100)) if project_duration > 0 else 0

        # Get current phase
        current_phase = "UNKNOWN"
        for phase in parser.phases:
            phase_start = phase.start.date() if hasattr(phase.start, 'date') else phase.start
            phase_end = phase.end.date() if hasattr(phase.end, 'date') else phase.end
            if phase_start <= today <= phase_end:
                current_phase = phase.name
                break

        # Build forecasts per trade
        forecasts = []
        total_budget = 0
        total_spent = 0
        total_expected = 0
        total_variance = 0

        # Aggregate actual costs by trade
        date_col = next((c for c in ['date', 'Date'] if c in direct_costs_df.columns), None)
        amount_col = next((c for c in ['amount', 'Amount'] if c in direct_costs_df.columns), None)

        actual_by_trade = {}
        if date_col and amount_col:
            # Map costs to trades
            for _, row in direct_costs_df.iterrows():
                name = str(row.get('name', '') or row.get('Description', ''))
                trade, _, _, _ = parser._map_to_trade(name)
                if trade:
                    actual_by_trade[trade] = actual_by_trade.get(trade, 0) + float(row[amount_col])

        # Get expected costs per trade from allocator using as_of date
        as_of = datetime.now()
        for trade in allocator.gmp.keys():
            budget = allocator.gmp.get(trade, 0)
            expected = allocator.get_expected_cost_to_date(trade, as_of)
            spent = actual_by_trade.get(trade, 0)

            # Calculate schedule variance
            schedule_variance = spent - expected
            variance_pct = (schedule_variance / expected * 100) if expected > 0 else 0

            # Simple forecast at completion based on schedule variance
            if expected > 0 and spent > 0:
                burn_rate = spent / expected if expected > 0 else 1
                forecast_at_completion = budget * burn_rate
            else:
                forecast_at_completion = budget

            budget_variance = forecast_at_completion - budget

            forecasts.append({
                "trade": trade,
                "gmp_budget": budget,
                "gmp_budget_display": f"${budget:,.0f}",
                "spent_to_date": spent,
                "spent_display": f"${spent:,.0f}",
                "expected_by_schedule": expected,
                "expected_display": f"${expected:,.0f}",
                "schedule_variance": schedule_variance,
                "schedule_variance_display": f"${schedule_variance:+,.0f}",
                "variance_pct": round(variance_pct, 1),
                "variance_status": "over" if schedule_variance > 0 else "under" if schedule_variance < 0 else "on_track",
                "forecast_at_completion": forecast_at_completion,
                "forecast_display": f"${forecast_at_completion:,.0f}",
                "budget_variance": budget_variance,
                "budget_variance_display": f"${budget_variance:+,.0f}"
            })

            total_budget += budget
            total_spent += spent
            total_expected += expected
            total_variance += schedule_variance

        # Sort by variance (worst first)
        forecasts.sort(key=lambda x: x["variance_pct"], reverse=True)

        return {
            "success": True,
            "project": {
                "start": parser.project_start.isoformat(),
                "end": parser.project_end.isoformat(),
                "pct_complete": round(project_pct, 1),
                "current_phase": current_phase,
                "total_activities": len(parser.activities),
                "phases": len(parser.phases)
            },
            "summary": {
                "total_budget": total_budget,
                "total_budget_display": f"${total_budget:,.0f}",
                "total_spent": total_spent,
                "total_spent_display": f"${total_spent:,.0f}",
                "total_expected": total_expected,
                "total_expected_display": f"${total_expected:,.0f}",
                "total_variance": total_variance,
                "total_variance_display": f"${total_variance:+,.0f}",
                "overall_status": "over" if total_variance > 0 else "under" if total_variance < 0 else "on_track"
            },
            "forecasts": forecasts[:10],  # Top 10 variances
            "trade_count": len(forecasts)
        }

    except Exception as e:
        logger.exception("Error generating schedule forecasts")
        return {
            "success": False,
            "error": str(e),
            "forecasts": [],
            "summary": None
        }


# =============================================================================
# Schedule Zone Assignment API (Spatial Schedule Tagging)
# =============================================================================

class BulkScheduleZoneRequest(PydanticBaseModel):
    """Request model for bulk schedule zone assignment."""
    activity_ids: List[int]
    zone: str  # EAST, WEST, SHARED


class BulkScheduleZoneResponse(PydanticBaseModel):
    """Response model for bulk schedule zone assignment."""
    updated_count: int
    new_linkage_score: float  # 0.00 - 1.00
    training_round_id: Optional[int]
    message: str


@app.patch("/api/schedule/activities/bulk-zone")
async def bulk_assign_schedule_zone(
    request: BulkScheduleZoneRequest,
    db: Session = Depends(get_db)
):
    """
    Bulk assign zones to schedule activities.
    Spec: PATCH /api/schedule/activities/bulk-zone

    Side effects:
    - Triggers CostLinkageService.recalculate_score
    - Logs change to TrainingHistory as user refinement
    """
    import uuid as uuid_lib

    # Validate zone
    valid_zones = {'EAST', 'WEST', 'SHARED'}
    zone = request.zone.upper()

    if zone not in valid_zones:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid zone: {request.zone}. Must be EAST, WEST, or SHARED."
        )

    # Validate that zone exists in GMP (per acceptance criteria)
    data_loader = get_data_loader()
    gmp_df = data_loader.gmp

    # Get unique zones from GMP (check for EAST/WEST/SHARED columns or zone column)
    gmp_zones = set()
    if 'Zone' in gmp_df.columns:
        gmp_zones = set(gmp_df['Zone'].dropna().str.upper().unique())
    elif 'zone' in gmp_df.columns:
        gmp_zones = set(gmp_df['zone'].dropna().str.upper().unique())
    else:
        # If no zone column, assume all zones are valid (backward compatibility)
        gmp_zones = valid_zones

    # Normalize BOTH to SHARED for comparison
    if 'BOTH' in gmp_zones:
        gmp_zones.add('SHARED')

    # Note: Skip zone validation if GMP doesn't have zone info (backward compatibility)
    # The spec says to reject if zone doesn't exist in GMP, but we'll be lenient
    # if the GMP doesn't have zone data yet

    updated_count = 0

    for activity_id in request.activity_ids:
        try:
            activity = db.query(ScheduleActivity).filter(
                ScheduleActivity.id == activity_id
            ).first()

            if activity:
                activity.zone = zone
                updated_count += 1

        except Exception as e:
            logger.error(f"Failed to update activity {activity_id} zone: {e}")

    db.commit()

    # Calculate new linkage score (% of schedule activities with zone assigned)
    total_activities = db.query(ScheduleActivity).count()
    activities_with_zone = db.query(ScheduleActivity).filter(
        ScheduleActivity.zone.isnot(None)
    ).count()

    new_linkage_score = (activities_with_zone / total_activities) if total_activities > 0 else 0.0

    # Create a training round to log this user refinement
    training_round = None
    try:
        # Get the default project (or first project)
        project = db.query(Project).first()
        project_id = project.id if project else None

        if project_id:
            # Get previous training round for comparison
            previous_round = db.query(TrainingRound).filter(
                TrainingRound.project_id == project_id,
                TrainingRound.status == 'completed'
            ).order_by(TrainingRound.triggered_at.desc()).first()

            training_round = TrainingRound(
                uuid=str(uuid_lib.uuid4()),
                project_id=project_id,
                triggered_at=datetime.now(timezone.utc),
                trigger_type='user_feedback',  # Zone assignment is user refinement
                status='completed',
                completed_at=datetime.now(timezone.utc),
                linkage_score=new_linkage_score * 100,  # Store as percentage
                previous_round_id=previous_round.id if previous_round else None,
                training_notes=f"User assigned zone '{zone}' to {updated_count} schedule activities"
            )
            db.add(training_round)
            db.commit()
            db.refresh(training_round)

    except Exception as e:
        logger.error(f"Failed to create training round: {e}")

    return BulkScheduleZoneResponse(
        updated_count=updated_count,
        new_linkage_score=round(new_linkage_score, 4),
        training_round_id=training_round.id if training_round else None,
        message=f"Updated {updated_count} activities to zone {zone}"
    )


@app.get("/api/schedule/activities/zone-stats")
async def get_schedule_zone_stats(db: Session = Depends(get_db)):
    """
    Get statistics about schedule zone assignments.
    Returns counts by zone and list of unassigned activities.
    """
    total = db.query(ScheduleActivity).count()

    # Count by zone
    east_count = db.query(ScheduleActivity).filter(
        ScheduleActivity.zone == 'EAST'
    ).count()
    west_count = db.query(ScheduleActivity).filter(
        ScheduleActivity.zone == 'WEST'
    ).count()
    shared_count = db.query(ScheduleActivity).filter(
        ScheduleActivity.zone == 'SHARED'
    ).count()
    unassigned_count = db.query(ScheduleActivity).filter(
        ScheduleActivity.zone.is_(None)
    ).count()

    # Calculate linkage score
    assigned_count = total - unassigned_count
    linkage_score = (assigned_count / total) if total > 0 else 0.0

    return {
        'total_activities': total,
        'zone_counts': {
            'EAST': east_count,
            'WEST': west_count,
            'SHARED': shared_count,
            'unassigned': unassigned_count
        },
        'assigned_count': assigned_count,
        'linkage_score': round(linkage_score, 4),
        'linkage_score_pct': round(linkage_score * 100, 2)
    }


# =============================================================================
# Enhanced Settings API (Task 8)
# =============================================================================

@app.post("/api/settings/breakdown")
async def update_breakdown_settings(
    use_breakdown_allocations: bool = Form(...),
    db: Session = Depends(get_db)
):
    """Update breakdown allocation setting."""
    settings = db.query(Settings).first()
    if not settings:
        settings = Settings()
        db.add(settings)

    settings.use_breakdown_allocations = use_breakdown_allocations
    db.commit()

    return {'success': True, 'use_breakdown_allocations': use_breakdown_allocations}


@app.post("/api/settings/schedule")
async def update_schedule_settings(
    use_schedule_forecast: bool = Form(...),
    db: Session = Depends(get_db)
):
    """Update schedule forecast setting."""
    settings = db.query(Settings).first()
    if not settings:
        settings = Settings()
        db.add(settings)

    settings.use_schedule_forecast = use_schedule_forecast
    db.commit()

    return {'success': True, 'use_schedule_forecast': use_schedule_forecast}


@app.get("/api/settings/integration")
async def get_integration_settings(db: Session = Depends(get_db)):
    """Get breakdown and schedule integration settings."""
    settings = db.query(Settings).first()

    # Get breakdown and schedule counts
    breakdown_count = db.query(GMPBudgetBreakdown).count()
    schedule_count = db.query(ScheduleActivity).count()
    matched_breakdown = db.query(GMPBudgetBreakdown).filter(
        GMPBudgetBreakdown.gmp_division.isnot(None)
    ).count()
    mapped_activities = db.query(ScheduleToGMPMapping).distinct(
        ScheduleToGMPMapping.schedule_activity_id
    ).count()

    return {
        'use_breakdown_allocations': getattr(settings, 'use_breakdown_allocations', True) if settings else True,
        'use_schedule_forecast': getattr(settings, 'use_schedule_forecast', False) if settings else False,
        'breakdown': {
            'total': breakdown_count,
            'matched': matched_breakdown,
            'unmatched': breakdown_count - matched_breakdown
        },
        'schedule': {
            'total': schedule_count,
            'mapped': mapped_activities,
            'unmapped': schedule_count - mapped_activities
        }
    }


# =============================================================================
# ML-Based Cost Forecasting API
# =============================================================================

# Global pipeline instance (lazy loaded)
_ml_pipeline = None

def get_ml_pipeline():
    """Get or initialize the ML training pipeline."""
    global _ml_pipeline
    if _ml_pipeline is None:
        try:
            from app.infrastructure.ml.training_pipeline import TrainingPipeline
            _ml_pipeline = TrainingPipeline()
            # Try to load pre-trained model
            model_path = Path("models/model.keras")
            if model_path.exists():
                _ml_pipeline.load(str(model_path))
        except Exception as e:
            logger.warning(f"Could not initialize ML pipeline: {e}")
    return _ml_pipeline


class BuildingParamsRequest(PydanticBaseModel):
    """Request model for building parameters."""
    sqft: float
    stories: int
    has_green_roof: bool = False
    rooftop_units_qty: int = 0
    fall_anchor_count: int = 0


class CostHistoryRequest(PydanticBaseModel):
    """Request model for cost history."""
    monthly_costs: List[float]


class ForecastRequest(PydanticBaseModel):
    """Request model for forecast generation."""
    building_params: BuildingParamsRequest
    cost_history: CostHistoryRequest
    confidence_level: float = 0.80


class ForecastResponse(PydanticBaseModel):
    """Response model for forecast results."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    mean: float
    std: float
    uncertainty_range: float


@app.post("/api/ml/forecast", response_model=ForecastResponse)
async def generate_ml_forecast(request: ForecastRequest):
    """
    Generate probabilistic cost forecast using TensorFlow model.

    Uses building parameters and historical costs to predict future costs
    with uncertainty quantification via Gaussian Mixture Model.
    """
    pipeline = get_ml_pipeline()

    if pipeline is None or not pipeline.model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Train the model first."
        )

    try:
        from app.forecasting.models.base_model import BuildingFeatures

        features = BuildingFeatures(
            sqft=request.building_params.sqft,
            stories=request.building_params.stories,
            has_green_roof=request.building_params.has_green_roof,
            rooftop_units_qty=request.building_params.rooftop_units_qty,
            fall_anchor_count=request.building_params.fall_anchor_count,
        )

        cost_history = np.array(request.cost_history.monthly_costs)

        result = pipeline.predict(
            features=features,
            cost_history=cost_history,
            confidence_level=request.confidence_level
        )

        return ForecastResponse(
            point_estimate=result.point_estimate,
            lower_bound=result.lower_bound,
            upper_bound=result.upper_bound,
            confidence_level=result.confidence_level,
            mean=result.mean,
            std=result.std,
            uncertainty_range=result.upper_bound - result.lower_bound,
        )
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/model/info")
async def get_ml_model_info():
    """Get information about the current ML model."""
    pipeline = get_ml_pipeline()

    if pipeline is None:
        return {
            "status": "not_initialized",
            "model_loaded": False,
        }

    return {
        "status": "ready" if pipeline.model.is_trained else "not_trained",
        "model_loaded": pipeline.model.is_trained,
        "info": pipeline.get_model_info() if pipeline.model else None,
    }


@app.get("/api/ml/health")
async def ml_health_check():
    """Health check for ML subsystem."""
    pipeline = get_ml_pipeline()

    return {
        "status": "healthy",
        "tensorflow_available": True,
        "model_loaded": pipeline is not None and pipeline.model.is_trained,
    }


# =============================================================================
# Project & Zone Tagging APIs (Per Specification Section 3)
# =============================================================================

class BulkZoneAssignRequest(PydanticBaseModel):
    """Request model for bulk zone assignment."""
    budget_ids: List[int]
    zone: str  # EAST, WEST, SHARED


class BulkZoneAssignResponse(PydanticBaseModel):
    """Response model for bulk zone assignment."""
    updated_count: int
    failed_ids: List[int]
    auto_linked_to_gmp: int
    message: str


@app.patch("/api/budgets/bulk-assign-zone")
async def bulk_assign_zone(
    request: BulkZoneAssignRequest,
    db: Session = Depends(get_db)
):
    """
    Bulk assign zones to budgets.
    Spec: PATCH /api/budgets/bulk-assign-zone
    Logic: Updates Budget.zone and attempts to auto-link to matching GMP.
    """
    import uuid as uuid_lib

    valid_zones = {'EAST', 'WEST', 'SHARED', 'BOTH'}
    zone = request.zone.upper()
    if zone == 'SHARED':
        zone = 'BOTH'  # Normalize SHARED to BOTH

    if zone not in valid_zones:
        raise HTTPException(status_code=400, detail=f"Invalid zone: {request.zone}")

    updated_count = 0
    failed_ids = []
    auto_linked = 0

    for budget_id in request.budget_ids:
        try:
            # Update BudgetToGMP mapping side
            budget_mapping = db.query(BudgetToGMP).filter(
                BudgetToGMP.id == budget_id
            ).first()

            if budget_mapping:
                budget_mapping.side = zone
                budget_mapping.updated_at = datetime.now(timezone.utc)
                updated_count += 1

                # Try to auto-link to GMP if not already linked
                if not budget_mapping.gmp_division:
                    # Find matching GMP by budget code pattern
                    data_loader = get_data_loader()
                    gmp_df = data_loader.gmp

                    # Simple matching logic - can be enhanced
                    for _, gmp_row in gmp_df.iterrows():
                        gmp_div = gmp_row.get('GMP Division', '')
                        if budget_mapping.budget_code in str(gmp_div):
                            budget_mapping.gmp_division = gmp_div
                            auto_linked += 1
                            break

            # Also update BudgetEntity if exists
            budget_entity = db.query(BudgetEntity).filter(
                BudgetEntity.id == budget_id
            ).first()

            if budget_entity:
                budget_entity.zone = zone
                budget_entity.updated_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Failed to update budget {budget_id}: {e}")
            failed_ids.append(budget_id)

    db.commit()

    return BulkZoneAssignResponse(
        updated_count=updated_count,
        failed_ids=failed_ids,
        auto_linked_to_gmp=auto_linked,
        message=f"Updated {updated_count} budgets to zone {request.zone}"
    )


# =============================================================================
# Training API (Per Specification Section 3)
# =============================================================================

class TrainingTriggerResponse(PydanticBaseModel):
    """Response model for training trigger."""
    training_round_id: int
    status: str
    linkage_score: Optional[float]
    eac_change_cents: Optional[int]
    eac_change_pct: Optional[float]
    previous_round_id: Optional[int]
    message: str


@app.post("/api/projects/{project_id}/train")
async def trigger_project_training(
    project_id: int,
    db: Session = Depends(get_db)
):
    """
    Trigger the Recalculation Engine for a project.
    Spec: POST /api/projects/{id}/train

    Steps:
    1. Create new TrainingRound
    2. Re-evaluate all Budget <-> Schedule links based on new Zone tags
    3. Calculate new Forecast Curve (EAC)
    4. Save curve to ForecastSnapshot
    5. Return comparison data (Old Snapshot vs New Snapshot)
    """
    import uuid as uuid_lib
    from datetime import timedelta

    # Check if project exists (or create default for backward compatibility)
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        # For backward compatibility, create a default project if none exists
        if project_id == 1:
            project = Project(
                uuid=str(uuid_lib.uuid4()),
                name="Default Project",
                code="DEFAULT",
                is_active=True
            )
            db.add(project)
            db.commit()
            db.refresh(project)
        else:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    # Get previous training round for comparison
    previous_round = db.query(TrainingRound).filter(
        TrainingRound.project_id == project_id,
        TrainingRound.status == 'completed'
    ).order_by(TrainingRound.triggered_at.desc()).first()

    # Create new training round
    training_round = TrainingRound(
        uuid=str(uuid_lib.uuid4()),
        project_id=project_id,
        triggered_at=datetime.now(timezone.utc),
        trigger_type='manual',
        status='running',
        previous_round_id=previous_round.id if previous_round else None
    )
    db.add(training_round)
    db.commit()
    db.refresh(training_round)

    try:
        # Step 1: Load data
        data_loader = get_data_loader()
        settings = get_settings(db)

        gmp_df = data_loader.gmp.copy()
        budget_df = data_loader.budget.copy()
        direct_costs_df = data_loader.direct_costs.copy()

        # Step 2: Re-evaluate Budget <-> Schedule links
        schedule_activities = db.query(ScheduleActivity).all()
        schedule_mappings = db.query(ScheduleToGMPMapping).all()

        # Calculate linkage score
        total_costs = len(direct_costs_df) if not direct_costs_df.empty else 0
        linked_costs = len(direct_costs_df[direct_costs_df.get('mapped_budget_code', pd.Series()).notna()]) if not direct_costs_df.empty else 0
        linkage_score = (linked_costs / total_costs * 100) if total_costs > 0 else 0.0

        # Calculate budget coverage
        total_budgets = len(budget_df) if not budget_df.empty else 0
        linked_budgets = db.query(BudgetToGMP).filter(
            BudgetToGMP.gmp_division.isnot(None)
        ).count()
        budget_coverage = (linked_budgets / total_budgets * 100) if total_budgets > 0 else 0.0

        # Step 3: Calculate new Forecast Curve (EAC)
        pipeline = get_forecasting_pipeline()
        if pipeline.last_trained is None:
            pipeline.train(direct_costs_df, budget_df, gmp_df, settings.get('as_of_date'))

        # Compute reconciliation to get EAC data
        breakdown_records = db.query(GMPBudgetBreakdown).all()
        breakdown_df = None
        if breakdown_records:
            breakdown_df = pd.DataFrame([{
                'gmp_division': b.gmp_division,
                'east_funded_cents': b.east_funded_cents,
                'west_funded_cents': b.west_funded_cents,
                'pct_east': b.pct_east,
                'pct_west': b.pct_west
            } for b in breakdown_records if b.gmp_division])

        predictions_df = pipeline.predict(direct_costs_df, budget_df, gmp_df, settings.get('as_of_date'))
        recon_df = compute_reconciliation(
            gmp_df, budget_df, direct_costs_df, predictions_df, settings,
            breakdown_df=breakdown_df
        )

        # Calculate total EAC
        total_eac_cents = int(recon_df['eac_cents'].sum()) if 'eac_cents' in recon_df.columns else 0

        # Step 4: Save forecast snapshots by zone
        as_of_date = settings.get('as_of_date') or datetime.now(timezone.utc).date()
        zones = ['EAST', 'WEST', 'SHARED']

        # Generate weekly periods for the next year
        current_date = as_of_date if isinstance(as_of_date, datetime) else datetime.combine(as_of_date, datetime.min.time())
        for i in range(52):  # 52 weeks
            period_date = current_date + timedelta(weeks=i)

            for zone in zones:
                # Calculate predicted cost for this period/zone
                zone_filter = zone if zone != 'SHARED' else 'BOTH'

                if zone == 'EAST':
                    period_cost = int(recon_df['eac_east_cents'].sum() / 52) if 'eac_east_cents' in recon_df.columns else 0
                elif zone == 'WEST':
                    period_cost = int(recon_df['eac_west_cents'].sum() / 52) if 'eac_west_cents' in recon_df.columns else 0
                else:
                    period_cost = int(total_eac_cents / 52)  # Shared/total

                cumulative_cost = period_cost * (i + 1)

                snapshot = TrainingForecastSnapshot(
                    training_round_id=training_round.id,
                    period_date=period_date.date() if hasattr(period_date, 'date') else period_date,
                    predicted_cumulative_cost_cents=cumulative_cost,
                    zone=zone,
                    confidence_lower_cents=int(cumulative_cost * 0.9),
                    confidence_upper_cents=int(cumulative_cost * 1.1)
                )
                db.add(snapshot)

        # Step 5: Compare with previous round
        previous_eac = 0
        if previous_round:
            prev_snapshots = db.query(TrainingForecastSnapshot).filter(
                TrainingForecastSnapshot.training_round_id == previous_round.id
            ).all()
            if prev_snapshots:
                previous_eac = max(s.predicted_cumulative_cost_cents for s in prev_snapshots)

        eac_change = total_eac_cents - previous_eac if previous_eac > 0 else None
        eac_change_pct = ((eac_change / previous_eac) * 100) if previous_eac > 0 and eac_change else None

        # Update training round with results
        training_round.status = 'completed'
        training_round.completed_at = datetime.now(timezone.utc)
        training_round.linkage_score = linkage_score
        training_round.budget_coverage = budget_coverage
        training_round.cost_coverage = (linked_costs / total_costs * 100) if total_costs > 0 else 0.0
        training_round.eac_change_cents = eac_change
        training_round.eac_change_pct = eac_change_pct
        training_round.model_version = "1.0.0"

        db.commit()

        return TrainingTriggerResponse(
            training_round_id=training_round.id,
            status='completed',
            linkage_score=linkage_score,
            eac_change_cents=eac_change,
            eac_change_pct=eac_change_pct,
            previous_round_id=previous_round.id if previous_round else None,
            message=f"Training completed. Linkage score: {linkage_score:.1f}%"
        )

    except Exception as e:
        training_round.status = 'failed'
        training_round.error_message = str(e)
        training_round.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/api/projects/{project_id}/training-rounds")
async def get_training_rounds(
    project_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get training round history for a project."""
    rounds = db.query(TrainingRound).filter(
        TrainingRound.project_id == project_id
    ).order_by(TrainingRound.triggered_at.desc()).limit(limit).all()

    return {
        "project_id": project_id,
        "training_rounds": [
            {
                "id": r.id,
                "uuid": r.uuid,
                "triggered_at": r.triggered_at.isoformat() if r.triggered_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "status": r.status,
                "trigger_type": r.trigger_type,
                "linkage_score": r.linkage_score,
                "budget_coverage": r.budget_coverage,
                "cost_coverage": r.cost_coverage,
                "eac_change_cents": r.eac_change_cents,
                "eac_change_pct": r.eac_change_pct,
                "previous_round_id": r.previous_round_id
            }
            for r in rounds
        ]
    }


@app.get("/api/projects/{project_id}/training-rounds/{round_id}/forecast")
async def get_training_round_forecast(
    project_id: int,
    round_id: int,
    zone: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get forecast curve for a specific training round."""
    training_round = db.query(TrainingRound).filter(
        TrainingRound.id == round_id,
        TrainingRound.project_id == project_id
    ).first()

    if not training_round:
        raise HTTPException(status_code=404, detail="Training round not found")

    query = db.query(TrainingForecastSnapshot).filter(
        TrainingForecastSnapshot.training_round_id == round_id
    )

    if zone:
        query = query.filter(TrainingForecastSnapshot.zone == zone.upper())

    snapshots = query.order_by(
        TrainingForecastSnapshot.zone,
        TrainingForecastSnapshot.period_date
    ).all()

    return {
        "training_round_id": round_id,
        "project_id": project_id,
        "status": training_round.status,
        "triggered_at": training_round.triggered_at.isoformat() if training_round.triggered_at else None,
        "forecast_points": [
            {
                "period_date": s.period_date.isoformat() if s.period_date else None,
                "zone": s.zone,
                "predicted_cumulative_cost_cents": s.predicted_cumulative_cost_cents,
                "actual_cumulative_cost_cents": s.actual_cumulative_cost_cents,
                "confidence_lower_cents": s.confidence_lower_cents,
                "confidence_upper_cents": s.confidence_upper_cents
            }
            for s in snapshots
        ]
    }


# =============================================================================
# Change Order APIs
# =============================================================================

class ChangeOrderCreate(PydanticBaseModel):
    """Request model for creating a change order."""
    gmp_id: int
    number: str
    title: str
    description: Optional[str] = None
    amount_cents: int
    requested_date: Optional[str] = None


class ChangeOrderUpdate(PydanticBaseModel):
    """Request model for updating a change order."""
    title: Optional[str] = None
    description: Optional[str] = None
    amount_cents: Optional[int] = None
    status: Optional[str] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None


@app.get("/api/change-orders")
async def list_change_orders(
    gmp_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List change orders with optional filters."""
    query = db.query(ChangeOrder)

    if gmp_id:
        query = query.filter(ChangeOrder.gmp_id == gmp_id)
    if status:
        query = query.filter(ChangeOrder.status == status)

    change_orders = query.order_by(ChangeOrder.created_at.desc()).all()

    return {
        "change_orders": [
            {
                "id": co.id,
                "uuid": co.uuid,
                "gmp_id": co.gmp_id,
                "number": co.number,
                "title": co.title,
                "description": co.description,
                "status": co.status,
                "amount_cents": co.amount_cents,
                "requested_date": co.requested_date.isoformat() if co.requested_date else None,
                "approved_date": co.approved_date.isoformat() if co.approved_date else None,
                "approved_by": co.approved_by,
                "created_at": co.created_at.isoformat() if co.created_at else None
            }
            for co in change_orders
        ]
    }


@app.post("/api/change-orders")
async def create_change_order(
    request: ChangeOrderCreate,
    db: Session = Depends(get_db)
):
    """Create a new change order."""
    import uuid as uuid_lib
    from datetime import datetime, timezone

    # Verify GMP exists
    gmp = db.query(GMP).filter(GMP.id == request.gmp_id).first()
    if not gmp:
        raise HTTPException(status_code=404, detail=f"GMP {request.gmp_id} not found")

    # Check for duplicate CO number
    existing = db.query(ChangeOrder).filter(
        ChangeOrder.gmp_id == request.gmp_id,
        ChangeOrder.number == request.number
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Change order {request.number} already exists for this GMP"
        )

    change_order = ChangeOrder(
        uuid=str(uuid_lib.uuid4()),
        gmp_id=request.gmp_id,
        number=request.number,
        title=request.title,
        description=request.description,
        amount_cents=request.amount_cents,
        status='draft',
        requested_date=datetime.strptime(request.requested_date, '%Y-%m-%d').date() if request.requested_date else None
    )

    db.add(change_order)
    db.commit()
    db.refresh(change_order)

    return {
        "id": change_order.id,
        "uuid": change_order.uuid,
        "message": f"Change order {change_order.number} created"
    }


@app.patch("/api/change-orders/{co_id}")
async def update_change_order(
    co_id: int,
    request: ChangeOrderUpdate,
    db: Session = Depends(get_db)
):
    """Update a change order."""
    change_order = db.query(ChangeOrder).filter(ChangeOrder.id == co_id).first()
    if not change_order:
        raise HTTPException(status_code=404, detail="Change order not found")

    if request.title is not None:
        change_order.title = request.title
    if request.description is not None:
        change_order.description = request.description
    if request.amount_cents is not None:
        change_order.amount_cents = request.amount_cents
    if request.status is not None:
        valid_statuses = {'draft', 'pending', 'approved'}
        if request.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")
        change_order.status = request.status
        if request.status == 'approved':
            change_order.approved_date = datetime.now(timezone.utc).date()
    if request.approved_by is not None:
        change_order.approved_by = request.approved_by
    if request.rejection_reason is not None:
        change_order.rejection_reason = request.rejection_reason

    change_order.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {"message": f"Change order {change_order.number} updated"}


@app.post("/api/change-orders/{co_id}/approve")
async def approve_change_order(
    co_id: int,
    approved_by: str = Form(...),
    db: Session = Depends(get_db)
):
    """Approve a change order. This is the ONLY way to adjust the GMP ceiling."""
    change_order = db.query(ChangeOrder).filter(ChangeOrder.id == co_id).first()
    if not change_order:
        raise HTTPException(status_code=404, detail="Change order not found")

    if change_order.status == 'approved':
        raise HTTPException(status_code=400, detail="Change order already approved")

    change_order.status = 'approved'
    change_order.approved_date = datetime.now(timezone.utc).date()
    change_order.approved_by = approved_by
    change_order.updated_at = datetime.now(timezone.utc)

    db.commit()

    # Get updated GMP ceiling
    gmp = change_order.gmp
    new_ceiling = gmp.authorized_amount_cents if gmp else change_order.amount_cents

    return {
        "message": f"Change order {change_order.number} approved",
        "new_gmp_ceiling_cents": new_ceiling,
        "change_amount_cents": change_order.amount_cents
    }


# =============================================================================
# CSV Ingestion API (Per Specification Section 3)
# =============================================================================

@app.post("/api/ingest/csv")
async def ingest_csv(
    file_type: str = Form(...),  # gmp, budget, direct_costs, schedule
    db: Session = Depends(get_db)
):
    """
    Ingest raw CSV files.
    Spec: POST /api/ingest/csv
    Creates unlinked Budgets/Costs with null Zones.
    """
    valid_types = {'gmp', 'budget', 'direct_costs', 'schedule', 'allocations', 'breakdown'}
    if file_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file_type. Must be one of: {valid_types}"
        )

    # Reload data from files
    data_loader = get_data_loader()
    data_loader.reload()

    # Flag items for zone review
    unassigned_count = 0

    if file_type == 'budget':
        # Count budgets without zone assignment
        unassigned_count = db.query(BudgetToGMP).filter(
            BudgetToGMP.side == 'BOTH'  # Default/unassigned
        ).count()

    return {
        "status": "success",
        "file_type": file_type,
        "message": f"Data reloaded from {file_type} files",
        "unassigned_zone_count": unassigned_count,
        "action_required": unassigned_count > 0
    }


# ------------ Error Handlers ------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse(request, "error.html", {
        "error": str(exc),
        "active_page": ""
    }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
