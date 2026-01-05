"""
Main FastAPI Application for GMP Reconciliation.
Serves HTML UI via Jinja2 templates and provides REST endpoints.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from typing import Dict
import json
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
import io

from app.models import (
    init_db, get_db, ensure_default_settings,
    Settings, Run, BudgetToGMP, DirectToBudget, Allocation, Duplicate
)
from app.modules.etl import get_data_loader, cents_to_display, get_file_hashes
from app.modules.mapping import (
    map_budget_to_gmp, map_direct_to_budget, apply_allocations,
    save_mapping, get_mapping_stats
)
from app.modules.reconciliation import (
    compute_reconciliation, format_for_display, compute_summary_metrics,
    validate_tie_outs, get_settings, get_gmp_drilldown
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


# Initialize FastAPI app
app = FastAPI(
    title="GMP Reconciliation App",
    description="Reconcile Procore Direct Costs against GMP funding via Budget mapping",
    version="1.0.0"
)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Custom Jinja2 filters
def format_currency(value):
    if isinstance(value, (int, float)):
        return cents_to_display(int(value))
    return value

templates.env.filters['currency'] = format_currency


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
            started_at=datetime.utcnow()
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
        run.finished_at = datetime.utcnow()
        run.notes = json.dumps(pipeline.get_training_status())
        db.commit()
    except Exception as e:
        run.status = 'failed'
        run.notes = str(e)
        run.finished_at = datetime.utcnow()
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

def run_full_reconciliation(db: Session) -> Dict:
    """
    Execute full reconciliation pipeline:
    1. Load data
    2. Run mappings
    3. Detect duplicates
    4. Apply allocations
    5. Run ML predictions
    6. Compute reconciliation
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
    
    # Detect duplicates
    duplicates, _ = detect_duplicates(direct_costs_df)
    direct_costs_df = apply_duplicate_exclusions(direct_costs_df, duplicates)
    
    # Apply allocations to budget (for commitments) - use base_code
    budget_df = apply_allocations(budget_df, 'committed_costs_cents', 'base_code', allocations_df, db)
    
    # Apply allocations to direct costs - use base_code
    direct_costs_df = apply_allocations(direct_costs_df, 'amount_cents', 'base_code', allocations_df, db)
    
    # Run ML predictions
    pipeline = get_forecasting_pipeline()
    if pipeline.last_trained is None:
        pipeline.train(direct_costs_df, budget_df, gmp_df, settings.get('as_of_date'))
    
    predictions_df = pipeline.predict(direct_costs_df, budget_df, gmp_df, settings.get('as_of_date'))
    
    # Compute reconciliation
    recon_df = compute_reconciliation(gmp_df, budget_df, direct_costs_df, predictions_df, settings)
    
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
    return RedirectResponse(url="/gmp")


@app.get("/gmp", response_class=HTMLResponse)
async def gmp_page(request: Request, db: Session = Depends(get_db)):
    """Main GMP reconciliation table page."""
    try:
        result = run_full_reconciliation(db)
        
        return templates.TemplateResponse("gmp.html", {
            "request": request,
            "rows": result['recon_rows'],
            "summary": result['summary'],
            "tie_outs": result['tie_outs'],
            "mapping_stats": result['mapping_stats'],
            "duplicates_summary": result['duplicates_summary'],
            "settings": result['settings'],
            "last_ml_train": result['last_ml_train'],
            "active_page": "gmp"
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "active_page": "gmp"
        })


@app.get("/gmp/{gmp_division}/forecast", response_class=HTMLResponse)
async def forecast_page(
    request: Request,
    gmp_division: str,
    granularity: str = "weekly",
    db: Session = Depends(get_db)
):
    """Forecast page for a specific GMP division."""
    from urllib.parse import unquote

    gmp_division = unquote(gmp_division)

    # Validate granularity
    if granularity not in ['weekly', 'monthly']:
        granularity = 'weekly'

    # Get forecast data
    manager = ForecastManager(db)
    config = manager.get_or_create_config(gmp_division)
    snapshot = manager.get_current_snapshot(gmp_division)

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
            'trigger': snapshot.trigger
        }
    else:
        forecast = {'has_forecast': False}

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

    return templates.TemplateResponse("forecast.html", {
        "request": request,
        "gmp_division": gmp_division,
        "granularity": granularity,
        "forecast": forecast,
        "config": config_dict,
        "periods": periods_data,
        "active_page": "gmp"
    })


@app.get("/mappings", response_class=HTMLResponse)
async def mappings_page(request: Request, tab: str = "budget_to_gmp", db: Session = Depends(get_db)):
    """Mapping editor page with tabs for Budget→GMP, Direct→Budget, and Allocations."""
    from rapidfuzz import fuzz, process

    data_loader = get_data_loader()

    # Get mappings from database
    budget_mappings_db = db.query(BudgetToGMP).all()
    direct_mappings_db = db.query(DirectToBudget).all()
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
    for m in budget_mappings_db:
        mapped_budget_codes.add(m.budget_code)
        budget_mappings.append({
            'budget_code': m.budget_code,
            'gmp_division': m.gmp_division,
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
    unmapped_budget = [b for b in all_budget_items if not b['is_mapped']]
    mapped_budget = [b for b in all_budget_items if b['is_mapped']]

    # Sort unmapped: items with suggestions first
    unmapped_budget.sort(key=lambda x: (-float(x.get('suggestion_confidence', 0) or 0), str(x.get('Budget Code', '') or '')))

    # Count suggestions
    suggested_budget = [b for b in unmapped_budget if b.get('suggested_gmp')]

    # Build direct cost mappings lookup from database
    direct_mappings_lookup = {}
    for m in direct_mappings_db:
        key = (m.cost_code, m.name)
        direct_mappings_lookup[key] = {
            'budget_code': m.budget_code,
            'confidence': m.confidence
        }

    # Compute suggestions for ALL direct costs (for unmapped ones)
    unmapped_direct_df = direct_df[direct_df['mapped_budget_code'].isna()].copy()
    dc_suggestions = compute_all_suggestions(
        unmapped_direct_df,
        data_loader.budget,
        db,
        unmapped_only=True,
        top_k=3
    )

    # Build ALL direct cost items list (both mapped and unmapped)
    all_direct_items = []
    display_columns = ['Cost Code', 'Name', 'Vendor', 'Invoice', 'Date', 'Amount', 'Type']

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
            item['mapped_budget_code'] = mapping['budget_code']
            item['mapping_confidence'] = mapping['confidence']
            item['budget_description'] = budget_desc_lookup.get(mapping['budget_code'], '')
            item['confidence_band'] = 'mapped'
            item['suggestions'] = []
            item['top_suggestion'] = None
        else:
            item['is_mapped'] = False
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
    unmapped_direct = [d for d in all_direct_items if not d['is_mapped']]
    mapped_direct = [d for d in all_direct_items if d['is_mapped']]

    # Sort unmapped by confidence (high first), mapped by cost code
    unmapped_direct.sort(key=lambda x: (-x.get('confidence', 0), str(x.get('Cost Code', ''))))
    mapped_direct.sort(key=lambda x: str(x.get('Cost Code', '')))

    # Limit unmapped for display (keep all mapped)
    unmapped_direct = unmapped_direct[:500]

    # Count by confidence band
    direct_high = len([d for d in unmapped_direct if d.get('confidence_band') == 'high'])
    direct_medium = len([d for d in unmapped_direct if d.get('confidence_band') == 'medium'])
    direct_low = len([d for d in unmapped_direct if d.get('confidence_band') == 'low'])

    return templates.TemplateResponse("mappings.html", {
        "request": request,
        "active_tab": tab,
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

    # Filter to unmapped only
    direct_df = map_direct_to_budget(direct_df, budget_df, db)
    unmapped_df = direct_df[direct_df['mapped_budget_code'].isna()].copy()

    if direct_cost_ids:
        unmapped_df = unmapped_df[unmapped_df['direct_cost_id'].isin(direct_cost_ids)]

    # Compute suggestions
    suggestions = compute_all_suggestions(unmapped_df, budget_df, db, top_k=1)

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
    
    return templates.TemplateResponse("duplicates.html", {
        "request": request,
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
        dup.resolved_at = datetime.utcnow()
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
    
    return templates.TemplateResponse("data_health.html", {
        "request": request,
        "issues": issues,
        "stats": stats,
        "active_page": "data-health"
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: Session = Depends(get_db)):
    """Settings page."""
    settings = db.query(Settings).first()
    data_loader = get_data_loader()
    
    # Get last run info
    last_run = db.query(Run).order_by(Run.started_at.desc()).first()
    
    return templates.TemplateResponse("settings.html", {
        "request": request,
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
    settings.updated_at = datetime.utcnow()
    
    db.commit()
    
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/recompute")
async def trigger_recompute(db: Session = Depends(get_db)):
    """Trigger full recomputation."""
    run = Run(
        run_type='recompute',
        status='running',
        started_at=datetime.utcnow(),
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
        run.finished_at = datetime.utcnow()
    except Exception as e:
        run.status = 'failed'
        run.notes = str(e)
        run.finished_at = datetime.utcnow()
    
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


@app.get("/api/gmp/drilldown/{gmp_division}")
async def gmp_drilldown(gmp_division: str, db: Session = Depends(get_db)):
    """
    Get detailed breakdown of direct costs for a GMP division.
    Shows budget codes and individual records contributing to Assigned West/East.
    """
    data_loader = get_data_loader()
    settings = get_settings(db)

    # Get data with mappings applied
    gmp_df = data_loader.gmp.copy()
    budget_df = map_budget_to_gmp(data_loader.budget.copy(), gmp_df, db)
    direct_df = map_direct_to_budget(data_loader.direct_costs.copy(), budget_df, db)

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
    General: { "distribution_method": "linear|front_loaded|back_loaded", "completion_date": "2026-06-30" }
    """
    body = await request.json()

    # Validate and convert date if present
    if 'completion_date' in body and body['completion_date']:
        try:
            body['completion_date'] = datetime.fromisoformat(body['completion_date'])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DD)")

    # Whitelist of allowed parameters
    allowed_params = {
        'evm_performance_factor', 'pert_optimistic_cents', 'pert_most_likely_cents',
        'pert_pessimistic_cents', 'param_quantity', 'param_unit_rate_cents',
        'param_complexity_factor', 'distribution_method', 'completion_date',
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
    as_of_date = settings.get('as_of_date') or loader.max_transaction_date or datetime.utcnow()
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)
    if hasattr(as_of_date, 'to_pydatetime'):
        as_of_date = as_of_date.to_pydatetime()

    if not division_direct.empty and 'Date' in division_direct.columns:
        start_date = pd.to_datetime(division_direct['Date'].min())
        if hasattr(start_date, 'to_pydatetime'):
            start_date = start_date.to_pydatetime()
    else:
        start_date = datetime(2025, 1, 1)

    config = manager.get_or_create_config(gmp_division)
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


# ------------ Error Handlers ------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error": str(exc),
        "active_page": ""
    }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
