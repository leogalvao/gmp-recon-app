"""
Direct Cost API Endpoints - CRUD operations for Direct Cost entities.

Implements:
- POST /api/v1/direct-costs - Create direct cost
- GET /api/v1/direct-costs/{id} - Get direct cost
- PATCH /api/v1/direct-costs/{id} - Update direct cost
- DELETE /api/v1/direct-costs/{id} - Delete direct cost
- POST /api/v1/direct-costs/bulk-map - Bulk map costs to budgets
"""
from typing import List, Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db
from app.infrastructure.repositories import DirectCostRepository, BudgetRepository
from app.domain.services import CostAggregationService
from app.domain.exceptions import (
    DirectCostNotFoundError,
    BudgetNotFoundError,
)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class DirectCostCreate(BaseModel):
    """Request model for creating a direct cost."""
    gross_amount_cents: int = Field(..., description="Cost amount in cents")
    vendor_name: Optional[str] = Field(None, max_length=255, description="Vendor name")
    description: Optional[str] = Field(None, max_length=500, description="Transaction description")
    transaction_date: Optional[date] = Field(None, description="Date of transaction")
    mapped_budget_id: Optional[int] = Field(None, description="Budget to map to")
    retainage_amount_cents: int = Field(0, ge=0, description="Retainage held back")
    allocation_method: str = Field("direct", pattern="^(direct|split_50_50)$")


class DirectCostUpdate(BaseModel):
    """Request model for updating a direct cost."""
    gross_amount_cents: Optional[int] = Field(None, description="Updated amount")
    vendor_name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    transaction_date: Optional[date] = Field(None)
    mapped_budget_id: Optional[int] = Field(None, description="New budget mapping (null to unmap)")
    retainage_amount_cents: Optional[int] = Field(None, ge=0)


class DirectCostResponse(BaseModel):
    """Response model for direct cost."""
    id: int
    uuid: str
    vendor_name: Optional[str]
    description: Optional[str]
    transaction_date: Optional[str]
    gross_amount_cents: int
    retainage_amount_cents: int
    payable_amount_cents: int
    mapped_budget_id: Optional[int]
    zone: Optional[str]
    allocation_method: str
    created_at: Optional[str]
    updated_at: Optional[str]

    class Config:
        from_attributes = True


class DirectCostListResponse(BaseModel):
    """Response for listing direct costs."""
    costs: List[DirectCostResponse]
    total: int
    total_amount_cents: int


class BulkMappingItem(BaseModel):
    """Single mapping in bulk operation."""
    direct_cost_id: int
    budget_id: Optional[int] = Field(None, description="Budget ID (null to unmap)")


class BulkMappingRequest(BaseModel):
    """Request for bulk mapping operation."""
    mappings: List[BulkMappingItem]


class BulkMappingResponse(BaseModel):
    """Response for bulk mapping operation."""
    updated_count: int
    affected_budgets: List[int]


class PeriodCostsResponse(BaseModel):
    """Response for period-aggregated costs."""
    period_id: str
    period_start: str
    period_end: str
    type: str  # actual, forecast, blended
    actual_cents: int
    forecast_cents: int
    total_cents: int


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "",
    response_model=DirectCostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new direct cost",
    description="Create a new direct cost transaction"
)
def create_direct_cost(
    cost_data: DirectCostCreate,
    db: Session = Depends(get_db)
):
    """Create a new direct cost entity."""
    repo = DirectCostRepository(db)
    aggregation_service = CostAggregationService(db)

    try:
        cost = repo.create(
            gross_amount_cents=cost_data.gross_amount_cents,
            vendor_name=cost_data.vendor_name,
            description=cost_data.description,
            transaction_date=cost_data.transaction_date,
            mapped_budget_id=cost_data.mapped_budget_id,
            retainage_amount_cents=cost_data.retainage_amount_cents,
            allocation_method=cost_data.allocation_method
        )
        db.commit()

        # Trigger cascade updates
        if cost.mapped_budget_id:
            aggregation_service.handle_direct_cost_change(
                cost_id=cost.id,
                operation='insert',
                new_budget_id=cost.mapped_budget_id
            )

        return _format_cost_response(cost)

    except BudgetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )


@router.get(
    "/{cost_id}",
    response_model=DirectCostResponse,
    summary="Get direct cost by ID"
)
def get_direct_cost(
    cost_id: int,
    db: Session = Depends(get_db)
):
    """Get a direct cost entity."""
    repo = DirectCostRepository(db)
    cost = repo.get_by_id(cost_id)

    if not cost:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Direct cost {cost_id} not found"
        )

    return _format_cost_response(cost)


@router.get(
    "",
    response_model=DirectCostListResponse,
    summary="List direct costs",
    description="List direct costs with optional filters"
)
def list_direct_costs(
    budget_id: Optional[int] = Query(None, description="Filter by budget"),
    unmapped_only: bool = Query(False, description="Show only unmapped costs"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date"),
    limit: int = Query(100, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """List direct costs with optional filters."""
    repo = DirectCostRepository(db)

    if unmapped_only:
        costs = repo.get_unmapped()
    elif budget_id:
        costs = repo.get_by_budget(budget_id)
    elif start_date and end_date:
        costs = repo.get_by_date_range(start_date, end_date)
    else:
        costs = repo.get_all(limit=limit, offset=offset)

    # Apply pagination to filtered results
    total = len(costs)
    costs = costs[offset:offset + limit]

    return {
        'costs': [_format_cost_response(c) for c in costs],
        'total': total,
        'total_amount_cents': sum(c.gross_amount_cents for c in costs)
    }


@router.patch(
    "/{cost_id}",
    response_model=DirectCostResponse,
    summary="Update a direct cost"
)
def update_direct_cost(
    cost_id: int,
    cost_data: DirectCostUpdate,
    db: Session = Depends(get_db)
):
    """Update a direct cost entity."""
    repo = DirectCostRepository(db)
    aggregation_service = CostAggregationService(db)

    cost = repo.get_by_id(cost_id)
    if not cost:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Direct cost {cost_id} not found"
        )

    old_budget_id = cost.mapped_budget_id

    # Update fields
    if cost_data.gross_amount_cents is not None:
        cost.gross_amount_cents = cost_data.gross_amount_cents
    if cost_data.vendor_name is not None:
        cost.vendor_name = cost_data.vendor_name
        cost.vendor_normalized = cost_data.vendor_name.lower().strip() if cost_data.vendor_name else None
    if cost_data.description is not None:
        cost.description = cost_data.description
    if cost_data.transaction_date is not None:
        cost.transaction_date = cost_data.transaction_date
    if cost_data.retainage_amount_cents is not None:
        cost.retainage_amount_cents = cost_data.retainage_amount_cents

    # Handle mapping change
    if 'mapped_budget_id' in cost_data.model_fields_set:
        try:
            cost, _ = repo.update_mapping(cost_id, cost_data.mapped_budget_id)
        except BudgetNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.message
            )

    db.commit()

    # Trigger cascade updates
    if old_budget_id or cost.mapped_budget_id:
        aggregation_service.handle_direct_cost_change(
            cost_id=cost.id,
            operation='update',
            old_budget_id=old_budget_id,
            new_budget_id=cost.mapped_budget_id
        )

    return _format_cost_response(cost)


@router.delete(
    "/{cost_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a direct cost"
)
def delete_direct_cost(
    cost_id: int,
    db: Session = Depends(get_db)
):
    """Delete a direct cost entity."""
    repo = DirectCostRepository(db)
    aggregation_service = CostAggregationService(db)

    cost = repo.get_by_id(cost_id)
    if not cost:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Direct cost {cost_id} not found"
        )

    old_budget_id = cost.mapped_budget_id

    repo.delete(cost)
    db.commit()

    # Trigger cascade updates
    if old_budget_id:
        aggregation_service.handle_direct_cost_change(
            cost_id=cost_id,
            operation='delete',
            old_budget_id=old_budget_id
        )


@router.post(
    "/bulk-map",
    response_model=BulkMappingResponse,
    summary="Bulk map direct costs to budgets",
    description="Map multiple direct costs to budgets in a single operation"
)
def bulk_map_costs(
    request: BulkMappingRequest,
    db: Session = Depends(get_db)
):
    """Bulk map direct costs to budgets."""
    repo = DirectCostRepository(db)

    mappings = [
        {'direct_cost_id': m.direct_cost_id, 'budget_id': m.budget_id}
        for m in request.mappings
    ]

    try:
        updated_count, affected_budgets = repo.bulk_map(mappings)
        db.commit()

        return {
            'updated_count': updated_count,
            'affected_budgets': affected_budgets
        }
    except BudgetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )


@router.get(
    "/by-budget/{budget_id}/periods",
    response_model=List[PeriodCostsResponse],
    summary="Get costs by period for a budget",
    description="Get costs aggregated by time period with actual/forecast split"
)
def get_period_costs(
    budget_id: int,
    start_date: date,
    end_date: date,
    granularity: str = Query("month", pattern="^(week|month)$"),
    db: Session = Depends(get_db)
):
    """Get costs aggregated by period for a budget."""
    aggregation_service = CostAggregationService(db)

    return aggregation_service.get_period_breakdown(
        start_date=start_date,
        end_date=end_date,
        as_of_date=date.today(),
        budget_id=budget_id,
        granularity=granularity
    )


@router.get(
    "/stats",
    summary="Get direct cost statistics",
    description="Get summary statistics for direct costs"
)
def get_cost_stats(
    budget_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get summary statistics for direct costs."""
    repo = DirectCostRepository(db)
    return repo.get_summary_stats(budget_id)


# =============================================================================
# Helper Functions
# =============================================================================

def _format_cost_response(cost) -> dict:
    """Format a direct cost entity for response."""
    return {
        'id': cost.id,
        'uuid': cost.uuid,
        'vendor_name': cost.vendor_name,
        'description': cost.description,
        'transaction_date': cost.transaction_date.isoformat() if cost.transaction_date else None,
        'gross_amount_cents': cost.gross_amount_cents,
        'retainage_amount_cents': cost.retainage_amount_cents,
        'payable_amount_cents': cost.payable_amount_cents,
        'mapped_budget_id': cost.mapped_budget_id,
        'zone': cost.zone,
        'allocation_method': cost.allocation_method,
        'created_at': cost.created_at.isoformat() if cost.created_at else None,
        'updated_at': cost.updated_at.isoformat() if cost.updated_at else None,
    }
