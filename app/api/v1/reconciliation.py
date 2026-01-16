"""
Reconciliation API Endpoints - Aggregation and earned value operations.

Implements:
- GET /api/v1/reconciliation/gmp/{gmp_id} - Full reconciliation for GMP
- GET /api/v1/reconciliation/evm/{gmp_division} - Earned value metrics
- GET /api/v1/reconciliation/periods - Period-based cost analysis
- GET /api/v1/reconciliation/validate - Validate reconciliation integrity
"""
from typing import List, Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db, User
from app.domain.services import (
    CostAggregationService,
    ScheduleLinkageService,
    BudgetValidationService,
)
from app.domain.exceptions import GMPNotFoundError
from app.api.v1.auth import get_current_active_user

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class DirectCostSummary(BaseModel):
    """Summary of a direct cost in reconciliation."""
    id: int
    uuid: str
    vendor_name: Optional[str]
    description: Optional[str]
    transaction_date: Optional[str]
    gross_amount_cents: int
    payable_amount_cents: int


class BudgetReconciliation(BaseModel):
    """Budget details in reconciliation."""
    id: int
    uuid: str
    cost_code: str
    zone: Optional[str]
    description: Optional[str]
    current_budget_cents: int
    committed_cents: int
    actual_cost_cents: int
    remaining_cents: int
    percent_spent: float
    direct_costs: List[DirectCostSummary]


class GMPReconciliation(BaseModel):
    """GMP details in reconciliation."""
    id: int
    uuid: str
    division: str
    zone: str
    description: Optional[str]
    original_amount_cents: int
    authorized_cents: int
    total_budgeted_cents: int
    total_actual_cents: int
    remaining_cents: int
    budget_utilization_pct: float
    spent_pct: float


class FullReconciliationResponse(BaseModel):
    """Complete reconciliation response."""
    gmp: GMPReconciliation
    budgets: List[BudgetReconciliation]
    unmapped_total_cents: int


class EVMResponse(BaseModel):
    """Earned Value Management metrics response."""
    gmp_division: str
    as_of_date: str
    bac_cents: int  # Budget at Completion
    pv_cents: int   # Planned Value
    ev_cents: int   # Earned Value
    ac_cents: int   # Actual Cost
    expected_pct_complete: float
    actual_pct_complete: float
    sv_cents: int   # Schedule Variance
    cv_cents: int   # Cost Variance
    sv_interpretation: str
    cv_interpretation: str
    spi: float      # Schedule Performance Index
    cpi: float      # Cost Performance Index
    spi_interpretation: str
    cpi_interpretation: str
    eac_cpi_cents: int        # EAC based on CPI
    eac_remaining_cents: int  # EAC based on remaining work
    eac_combined_cents: int   # EAC combined method
    etc_cents: int            # Estimate to Complete
    vac_cents: int            # Variance at Completion


class PeriodTotalResponse(BaseModel):
    """Period total in aggregation."""
    period_id: str
    period_start: str
    period_end: str
    amount_cents: int
    type: Optional[str] = None  # actual, forecast, blended


class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: List[str]


class ValidationSummaryResponse(BaseModel):
    """Validation summary for GMP."""
    gmp_id: int
    ceiling_cents: int
    total_budgeted_cents: int
    available_cents: int
    ceiling_utilization_pct: float
    budget_count: int
    is_ceiling_exceeded: bool
    warnings: List[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/gmp/{gmp_id}",
    response_model=FullReconciliationResponse,
    summary="Get full reconciliation for a GMP",
    description="Get complete hierarchical reconciliation: GMP → Budgets → Direct Costs"
)
def get_full_reconciliation(
    gmp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get complete reconciliation for a GMP."""
    aggregation_service = CostAggregationService(db)

    try:
        return aggregation_service.get_full_reconciliation(gmp_id)
    except GMPNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.get(
    "/evm/{gmp_division}",
    response_model=EVMResponse,
    summary="Get earned value metrics for a GMP division",
    description="Calculate full EVM metrics including PV, EV, AC, SV, CV, SPI, CPI, and EAC"
)
def get_evm_metrics(
    gmp_division: str,
    as_of_date: Optional[date] = Query(None, description="Reference date (default: today)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get earned value management metrics."""
    schedule_service = ScheduleLinkageService(db)
    return schedule_service.calculate_full_evm(gmp_division, as_of_date)


@router.get(
    "/gmp/{gmp_id}/weekly",
    response_model=List[PeriodTotalResponse],
    summary="Get weekly cost totals for a GMP",
    description="Aggregate costs by ISO week"
)
def get_weekly_totals(
    gmp_id: int,
    start_date: date,
    end_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get weekly cost totals for a GMP."""
    aggregation_service = CostAggregationService(db)

    # Note: This aggregates across all budgets under the GMP
    # For per-budget, use the direct-costs endpoint
    return aggregation_service.get_weekly_totals(start_date, end_date)


@router.get(
    "/gmp/{gmp_id}/monthly",
    response_model=List[PeriodTotalResponse],
    summary="Get monthly cost totals for a GMP",
    description="Aggregate costs by calendar month"
)
def get_monthly_totals(
    gmp_id: int,
    start_date: date,
    end_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get monthly cost totals for a GMP."""
    aggregation_service = CostAggregationService(db)
    return aggregation_service.get_monthly_totals(start_date, end_date)


@router.get(
    "/gmp/{gmp_id}/periods",
    response_model=List[PeriodTotalResponse],
    summary="Get period breakdown with actual/forecast split",
    description="Get costs by period with actual/forecast/blended classification"
)
def get_period_breakdown(
    gmp_id: int,
    start_date: date,
    end_date: date,
    granularity: str = Query("month", pattern="^(week|month)$"),
    as_of_date: Optional[date] = Query(None, description="Reference date for split"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get period breakdown with actual/forecast split."""
    aggregation_service = CostAggregationService(db)

    return aggregation_service.get_period_breakdown(
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date or date.today(),
        granularity=granularity
    )


@router.get(
    "/gmp/{gmp_id}/cumulative",
    summary="Get cumulative cost as of a date",
    description="Get total cumulative cost up to a specific date"
)
def get_cumulative_cost(
    gmp_id: int,
    as_of_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get cumulative cost as of a date."""
    aggregation_service = CostAggregationService(db)

    cumulative = aggregation_service.get_cumulative_actual(as_of_date, gmp_id=gmp_id)

    return {
        'gmp_id': gmp_id,
        'as_of_date': as_of_date.isoformat(),
        'cumulative_actual_cents': cumulative
    }


@router.get(
    "/gmp/{gmp_id}/validate",
    response_model=ValidationResult,
    summary="Validate vertical reconciliation",
    description="Check that all reconciliation invariants hold"
)
def validate_reconciliation(
    gmp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Validate reconciliation integrity."""
    aggregation_service = CostAggregationService(db)
    is_valid, errors = aggregation_service.validate_vertical_reconciliation(gmp_id)

    return {
        'is_valid': is_valid,
        'errors': errors
    }


@router.get(
    "/gmp/{gmp_id}/validation-summary",
    response_model=ValidationSummaryResponse,
    summary="Get validation summary for a GMP",
    description="Get detailed validation status including ceiling and budget health"
)
def get_validation_summary(
    gmp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get validation summary for a GMP."""
    validation_service = BudgetValidationService(db)

    try:
        return validation_service.get_validation_summary(gmp_id)
    except GMPNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.get(
    "/unmapped",
    summary="Get unmapped cost summary",
    description="Get total and count of unmapped direct costs"
)
def get_unmapped_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get summary of unmapped costs."""
    aggregation_service = CostAggregationService(db)
    unmapped_total = aggregation_service.get_unmapped_total()

    # Get count
    from app.infrastructure.repositories import DirectCostRepository
    cost_repo = DirectCostRepository(db)
    unmapped_costs = cost_repo.get_unmapped()

    return {
        'unmapped_count': len(unmapped_costs),
        'unmapped_total_cents': unmapped_total
    }


@router.get(
    "/schedule/{gmp_division}/forecast",
    summary="Get schedule-based forecast for a GMP division",
    description="Generate cost forecast based on schedule progress"
)
def get_schedule_forecast(
    gmp_division: str,
    start_date: date,
    end_date: date,
    granularity: str = Query("month", pattern="^(week|month)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get schedule-based cost forecast."""
    schedule_service = ScheduleLinkageService(db)

    return schedule_service.get_schedule_based_forecast(
        gmp_division=gmp_division,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity
    )


@router.post(
    "/schedule/recalculate",
    summary="Recalculate all forecasts",
    description="Trigger recalculation of all schedule-based forecasts"
)
def recalculate_forecasts(
    project_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Recalculate all forecasts."""
    schedule_service = ScheduleLinkageService(db)
    return schedule_service.recalculate_all_forecasts(project_id)


@router.get(
    "/prorate-week",
    summary="Calculate week proration across months",
    description="Calculate how a week's cost should be split across months"
)
def prorate_week(
    week_start: date,
    week_end: date,
    amount_cents: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Calculate week proration across months."""
    aggregation_service = CostAggregationService(db)

    return aggregation_service.prorate_spanning_week(
        week_start=week_start,
        week_end=week_end,
        week_amount_cents=amount_cents
    )
