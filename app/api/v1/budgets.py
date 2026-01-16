"""
Budget API Endpoints - CRUD operations for Budget entities.

Implements:
- POST /api/v1/gmp/{gmp_id}/budgets - Create budget under GMP
- GET /api/v1/budgets/{id} - Get budget with computed fields
- PATCH /api/v1/budgets/{id} - Update budget
- DELETE /api/v1/budgets/{id} - Delete budget (if no mapped costs)
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.models import get_db, User
from app.infrastructure.repositories import BudgetRepository, GMPRepository
from app.domain.services import BudgetValidationService, CostAggregationService
from app.domain.exceptions import (
    BudgetNotFoundError,
    GMPNotFoundError,
    GMPCeilingExceededError,
    BudgetUnderflowError,
    BudgetHasMappedCostsError,
)
from app.api.v1.auth import get_current_active_user

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class BudgetCreate(BaseModel):
    """Request model for creating a budget."""
    cost_code: str = Field(..., min_length=1, max_length=50, description="CSI cost code")
    current_budget_cents: int = Field(..., gt=0, description="Budget amount in cents")
    zone: Optional[str] = Field(None, pattern="^(EAST|WEST|SHARED)$", description="Zone assignment")
    description: Optional[str] = Field(None, max_length=500, description="Budget description")
    committed_cents: int = Field(0, ge=0, description="Committed amount in cents")


class BudgetUpdate(BaseModel):
    """Request model for updating a budget."""
    current_budget_cents: Optional[int] = Field(None, ge=0, description="New budget amount")
    description: Optional[str] = Field(None, max_length=500, description="Updated description")
    committed_cents: Optional[int] = Field(None, ge=0, description="Committed amount")
    zone: Optional[str] = Field(None, pattern="^(EAST|WEST|SHARED)$", description="Zone assignment")


class BudgetResponse(BaseModel):
    """Response model for budget with computed fields."""
    id: int
    uuid: str
    gmp_id: int
    cost_code: str
    zone: Optional[str]
    description: Optional[str]
    current_budget_cents: int
    committed_cents: int
    actual_cost_cents: int
    remaining_cents: int
    percent_spent: float
    mapped_cost_count: int
    created_at: Optional[str]
    updated_at: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class BudgetListResponse(BaseModel):
    """Response for listing budgets."""
    budgets: List[BudgetResponse]
    total: int
    total_budgeted_cents: int
    total_actual_cents: int


class BudgetTransferRequest(BaseModel):
    """Request model for budget transfer."""
    from_budget_id: int = Field(..., description="Source budget ID")
    to_budget_id: int = Field(..., description="Destination budget ID")
    amount_cents: int = Field(..., gt=0, description="Amount to transfer")


class BudgetTransferResponse(BaseModel):
    """Response for budget transfer."""
    from_budget: BudgetResponse
    to_budget: BudgetResponse
    transferred_cents: int


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/gmp/{gmp_id}/budgets",
    response_model=BudgetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new budget under a GMP",
    description="Create a new budget allocation. Validates GMP ceiling constraint."
)
def create_budget(
    gmp_id: int,
    budget_data: BudgetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new budget entity under a GMP.

    Validates:
    - GMP exists
    - GMP ceiling is not exceeded
    """
    validation_service = BudgetValidationService(db)
    repo = BudgetRepository(db)

    # Validate
    is_valid, errors = validation_service.validate_budget_create(
        gmp_id=gmp_id,
        amount_cents=budget_data.current_budget_cents,
        cost_code=budget_data.cost_code
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors}
        )

    try:
        budget = repo.create(
            gmp_id=gmp_id,
            cost_code=budget_data.cost_code,
            current_budget_cents=budget_data.current_budget_cents,
            zone=budget_data.zone,
            description=budget_data.description,
            committed_cents=budget_data.committed_cents
        )
        db.commit()

        return repo.get_summary(budget.id)

    except GMPNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except GMPCeilingExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )


@router.get(
    "/{budget_id}",
    response_model=BudgetResponse,
    summary="Get budget by ID",
    description="Retrieve a budget with computed fields"
)
def get_budget(
    budget_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a budget entity with computed fields."""
    repo = BudgetRepository(db)

    try:
        return repo.get_summary(budget_id)
    except BudgetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.get(
    "/gmp/{gmp_id}",
    response_model=BudgetListResponse,
    summary="List budgets for GMP",
    description="Get all budgets under a GMP with totals"
)
def list_budgets_for_gmp(
    gmp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all budgets for a GMP."""
    repo = BudgetRepository(db)
    gmp_repo = GMPRepository(db)

    # Verify GMP exists
    if not gmp_repo.get_by_id(gmp_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GMP {gmp_id} not found"
        )

    budgets = repo.get_all_for_gmp_with_actuals(gmp_id)

    total_budgeted = sum(b['current_budget_cents'] for b in budgets)
    total_actual = sum(b['actual_cost_cents'] for b in budgets)

    return {
        'budgets': budgets,
        'total': len(budgets),
        'total_budgeted_cents': total_budgeted,
        'total_actual_cents': total_actual
    }


@router.patch(
    "/{budget_id}",
    response_model=BudgetResponse,
    summary="Update a budget",
    description="Update budget fields. Validates ceiling and underflow constraints."
)
def update_budget(
    budget_id: int,
    budget_data: BudgetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a budget entity.

    Validates:
    - Cannot exceed GMP ceiling
    - Cannot reduce below actual spent
    """
    validation_service = BudgetValidationService(db)
    repo = BudgetRepository(db)

    # Get current budget
    budget = repo.get_by_id(budget_id)
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget {budget_id} not found"
        )

    # Validate amount change if provided
    if budget_data.current_budget_cents is not None:
        is_valid, errors = validation_service.validate_budget_update(
            budget_id=budget_id,
            new_amount_cents=budget_data.current_budget_cents
        )

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"errors": errors}
            )

        budget.current_budget_cents = budget_data.current_budget_cents

    # Update other fields
    if budget_data.description is not None:
        budget.description = budget_data.description
    if budget_data.committed_cents is not None:
        budget.committed_cents = budget_data.committed_cents
    if budget_data.zone is not None:
        budget.zone = budget_data.zone

    db.commit()

    return repo.get_summary(budget_id)


@router.delete(
    "/{budget_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a budget",
    description="Delete a budget. Fails if direct costs are mapped to it."
)
def delete_budget(
    budget_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a budget entity.

    Validates:
    - No direct costs are mapped to this budget
    """
    validation_service = BudgetValidationService(db)
    repo = BudgetRepository(db)

    # Validate
    is_valid, errors = validation_service.validate_budget_delete(budget_id)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"errors": errors}
        )

    try:
        repo.delete_budget(budget_id)
        db.commit()
    except BudgetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except BudgetHasMappedCostsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )


@router.post(
    "/transfer",
    response_model=BudgetTransferResponse,
    summary="Transfer budget between allocations",
    description="Transfer budget from one budget line to another under the same GMP"
)
def transfer_budget(
    transfer_data: BudgetTransferRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Transfer budget between two budget lines.

    Validates:
    - Both budgets exist
    - Same parent GMP
    - Source has enough budget above actual
    """
    validation_service = BudgetValidationService(db)
    repo = BudgetRepository(db)

    # Validate transfer
    is_valid, errors = validation_service.validate_budget_transfer(
        from_budget_id=transfer_data.from_budget_id,
        to_budget_id=transfer_data.to_budget_id,
        amount_cents=transfer_data.amount_cents
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors}
        )

    # Perform transfer
    from_budget = repo.get_by_id(transfer_data.from_budget_id)
    to_budget = repo.get_by_id(transfer_data.to_budget_id)

    from_budget.current_budget_cents -= transfer_data.amount_cents
    to_budget.current_budget_cents += transfer_data.amount_cents

    db.commit()

    return {
        'from_budget': repo.get_summary(transfer_data.from_budget_id),
        'to_budget': repo.get_summary(transfer_data.to_budget_id),
        'transferred_cents': transfer_data.amount_cents
    }


@router.get(
    "/unmapped-zone",
    response_model=List[BudgetResponse],
    summary="Get budgets without zone assignment",
    description="Get all budgets that haven't been assigned to a zone"
)
def get_unmapped_budgets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get budgets without zone assignment."""
    repo = BudgetRepository(db)
    budgets = repo.get_unmapped_to_zone()
    return [repo.get_summary(b.id) for b in budgets]


@router.post(
    "/bulk-zone",
    summary="Bulk update zone for budgets",
    description="Assign zone to multiple budgets at once"
)
def bulk_update_zone(
    budget_ids: List[int],
    zone: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Bulk update zone for multiple budgets."""
    if zone not in ('EAST', 'WEST', 'SHARED'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Zone must be EAST, WEST, or SHARED"
        )

    repo = BudgetRepository(db)
    count = repo.bulk_update_zone(budget_ids, zone)
    db.commit()

    return {'updated_count': count}
