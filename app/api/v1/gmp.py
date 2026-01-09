"""
GMP API Endpoints - CRUD operations for GMP entities.

Implements:
- POST /api/v1/gmp - Create GMP (amount is immutable after creation)
- GET /api/v1/gmp/{id} - Get GMP with computed fields
- GET /api/v1/projects/{project_id}/gmp - List GMPs for project
- PATCH /api/v1/gmp/{id} - Update description only (amount is immutable)
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db
from app.infrastructure.repositories import GMPRepository
from app.domain.services import BudgetValidationService
from app.domain.exceptions import (
    GMPNotFoundError,
    DuplicateGMPError,
    ImmutableFieldError,
)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class GMPCreate(BaseModel):
    """Request model for creating a GMP."""
    project_id: int = Field(..., description="Parent project ID")
    division: str = Field(..., min_length=1, max_length=200, description="CSI division name")
    zone: str = Field(..., pattern="^(EAST|WEST|SHARED)$", description="Zone: EAST, WEST, or SHARED")
    original_amount_cents: int = Field(..., gt=0, description="GMP amount in cents (immutable)")
    description: Optional[str] = Field(None, max_length=500, description="GMP description")


class GMPUpdate(BaseModel):
    """Request model for updating a GMP (description only)."""
    description: str = Field(..., max_length=500, description="Updated description")


class GMPResponse(BaseModel):
    """Response model for GMP with computed fields."""
    id: int
    uuid: str
    project_id: int
    division: str
    zone: str
    description: Optional[str]
    original_amount_cents: int
    change_order_total_cents: int
    authorized_amount_cents: int
    total_budgeted_cents: int
    total_actual_cents: int
    remaining_cents: int
    budget_utilization_pct: float
    spent_pct: float
    created_at: Optional[str]
    updated_at: Optional[str]

    class Config:
        from_attributes = True


class GMPSummaryResponse(BaseModel):
    """Lightweight GMP response for listings."""
    id: int
    uuid: str
    division: str
    zone: str
    authorized_amount_cents: int
    total_actual_cents: int
    spent_pct: float


class GMPListResponse(BaseModel):
    """Response for listing GMPs."""
    gmps: List[GMPSummaryResponse]
    total: int


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "",
    response_model=GMPResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new GMP",
    description="Create a new Guaranteed Maximum Price allocation. "
                "The amount is IMMUTABLE after creation."
)
def create_gmp(
    gmp_data: GMPCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new GMP entity.

    The original_amount_cents CANNOT be modified after creation.
    Use Change Orders to adjust the GMP ceiling.
    """
    repo = GMPRepository(db)

    try:
        gmp = repo.create(
            project_id=gmp_data.project_id,
            division=gmp_data.division,
            zone=gmp_data.zone,
            original_amount_cents=gmp_data.original_amount_cents,
            description=gmp_data.description
        )
        db.commit()

        return repo.get_summary(gmp.id)

    except DuplicateGMPError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{gmp_id}",
    response_model=GMPResponse,
    summary="Get GMP by ID",
    description="Retrieve a GMP with all computed fields"
)
def get_gmp(
    gmp_id: int,
    db: Session = Depends(get_db)
):
    """Get a GMP entity with computed fields."""
    repo = GMPRepository(db)

    try:
        return repo.get_summary(gmp_id)
    except GMPNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.get(
    "/project/{project_id}",
    response_model=GMPListResponse,
    summary="List GMPs for project",
    description="Get all GMPs for a project with totals"
)
def list_gmps_for_project(
    project_id: int,
    include_totals: bool = True,
    db: Session = Depends(get_db)
):
    """List all GMPs for a project."""
    repo = GMPRepository(db)

    if include_totals:
        gmps = repo.get_all_with_totals(project_id)
    else:
        raw_gmps = repo.get_by_project(project_id)
        gmps = [
            {
                'id': g.id,
                'uuid': g.uuid,
                'division': g.division,
                'zone': g.zone,
                'authorized_amount_cents': g.authorized_amount_cents,
                'total_actual_cents': 0,
                'spent_pct': 0,
            }
            for g in raw_gmps
        ]

    return {
        'gmps': gmps,
        'total': len(gmps)
    }


@router.patch(
    "/{gmp_id}",
    response_model=GMPResponse,
    summary="Update GMP description",
    description="Update GMP description. Amount CANNOT be modified."
)
def update_gmp_description(
    gmp_id: int,
    gmp_data: GMPUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a GMP's description.

    Note: The amount field is IMMUTABLE and cannot be changed.
    Use Change Orders to modify the GMP ceiling.
    """
    repo = GMPRepository(db)

    try:
        repo.update_description(gmp_id, gmp_data.description)
        db.commit()
        return repo.get_summary(gmp_id)

    except GMPNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.get(
    "/{gmp_id}/validation",
    summary="Get GMP validation summary",
    description="Get validation status including ceiling utilization and budget health"
)
def get_gmp_validation(
    gmp_id: int,
    db: Session = Depends(get_db)
):
    """Get validation summary for a GMP and its budgets."""
    validation_service = BudgetValidationService(db)

    try:
        return validation_service.get_validation_summary(gmp_id)
    except GMPNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


# Note: No DELETE endpoint - GMP cannot be deleted, only archived
# Note: No PUT endpoint for amount - GMP amount is immutable
