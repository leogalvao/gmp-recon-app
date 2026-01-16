"""
Migration API Endpoints - Phase 2 multi-project migration operations.

Implements:
- GET /api/v1/migration/canonical-trades - List canonical trades
- POST /api/v1/migration/suggest-mapping - Suggest trade mappings
- POST /api/v1/migration/map-division - Map a division to canonical trade
- POST /api/v1/migration/migrate-project - Migrate a project
- POST /api/v1/migration/migrate-all - Migrate all projects
- GET /api/v1/migration/status/{project_id} - Get migration status
- POST /api/v1/migration/backfill-features - Backfill feature store
"""
from typing import List, Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db, User, CanonicalTrade
from app.domain.services import (
    TradeMappingService,
    ProjectMigrationService,
    FeatureStoreService,
)
from app.api.v1.auth import get_current_active_user

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class CanonicalTradeResponse(BaseModel):
    """Response model for canonical trade."""
    id: int
    canonical_code: str
    csi_division: str
    canonical_name: str
    hierarchy_level: int
    typical_pct_of_total: Optional[float]
    typical_duration_pct: Optional[float]
    is_active: bool


class TradeMappingSuggestionResponse(BaseModel):
    """Response model for trade mapping suggestion."""
    canonical_trade_id: int
    canonical_code: str
    canonical_name: str
    confidence: float
    method: str
    match_reason: str


class SuggestMappingRequest(BaseModel):
    """Request model for suggesting trade mappings."""
    raw_division_name: str = Field(..., description="Raw division name to map")
    top_n: int = Field(3, ge=1, le=10, description="Number of suggestions")


class MapDivisionRequest(BaseModel):
    """Request model for mapping a division."""
    project_id: int
    raw_division_name: str
    canonical_trade_id: Optional[int] = Field(None, description="Manual mapping (null for auto)")


class TradeMappingResponse(BaseModel):
    """Response model for trade mapping result."""
    raw_division_name: str
    canonical_trade_id: int
    canonical_code: str
    canonical_name: str
    confidence: float
    mapping_method: str


class MigrateProjectRequest(BaseModel):
    """Request model for project migration."""
    project_id: int
    auto_confirm_threshold: float = Field(0.9, ge=0.0, le=1.0)
    force_remigrate: bool = False


class ProjectMigrationResponse(BaseModel):
    """Response model for project migration result."""
    project_id: int
    project_code: str
    success: bool
    trades_mapped: int
    trades_auto_confirmed: int
    trades_need_review: int
    square_feet_inferred: Optional[int]
    data_quality_score: float
    errors: List[str]
    warnings: List[str]


class MigrationStatusResponse(BaseModel):
    """Response model for migration status."""
    project_id: int
    project_code: str
    is_migrated: bool
    total_gmps: int
    gmps_with_canonical_trade: int
    trade_mappings_count: int
    mappings_need_review: int
    has_square_feet: bool
    square_feet: Optional[int]
    project_type: Optional[str]
    data_quality_score: Optional[float]
    is_training_eligible: Optional[bool]


class BackfillFeaturesRequest(BaseModel):
    """Request model for feature backfill."""
    project_id: Optional[int] = Field(None, description="Specific project (null for all)")
    period_type: str = Field('monthly', pattern='^(weekly|monthly)$')
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class FeatureBackfillResponse(BaseModel):
    """Response model for feature backfill result."""
    project_id: int
    periods_created: int
    trades_covered: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    errors: List[str]


class FullMigrationResponse(BaseModel):
    """Response model for full migration result."""
    total_projects: int
    successful: int
    failed: int
    total_trades_mapped: int
    project_results: List[ProjectMigrationResponse]


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/canonical-trades",
    response_model=List[CanonicalTradeResponse],
    summary="List all canonical trades",
    description="Get the CSI-based canonical trade taxonomy"
)
def list_canonical_trades(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all active canonical trades."""
    service = TradeMappingService(db)
    trades = service.get_all_canonical_trades()

    return [
        {
            'id': t.id,
            'canonical_code': t.canonical_code,
            'csi_division': t.csi_division,
            'canonical_name': t.canonical_name,
            'hierarchy_level': t.hierarchy_level,
            'typical_pct_of_total': t.typical_pct_of_total,
            'typical_duration_pct': t.typical_duration_pct,
            'is_active': t.is_active,
        }
        for t in trades
    ]


@router.post(
    "/suggest-mapping",
    response_model=List[TradeMappingSuggestionResponse],
    summary="Suggest trade mappings",
    description="Get suggested canonical trade mappings for a raw division name"
)
def suggest_mapping(
    request: SuggestMappingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Suggest canonical trade mappings for a raw division name."""
    service = TradeMappingService(db)
    suggestions = service.suggest_mapping(
        raw_division_name=request.raw_division_name,
        top_n=request.top_n
    )

    return [
        {
            'canonical_trade_id': s.canonical_trade.id,
            'canonical_code': s.canonical_trade.canonical_code,
            'canonical_name': s.canonical_trade.canonical_name,
            'confidence': s.confidence,
            'method': s.method,
            'match_reason': s.match_reason,
        }
        for s in suggestions
    ]


@router.post(
    "/map-division",
    response_model=TradeMappingResponse,
    summary="Map a division to canonical trade",
    description="Create or update a trade mapping for a GMP division"
)
def map_division(
    request: MapDivisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Map a GMP division to a canonical trade."""
    service = TradeMappingService(db)

    try:
        result = service.map_division(
            project_id=request.project_id,
            raw_division_name=request.raw_division_name,
            canonical_trade_id=request.canonical_trade_id,
            created_by=current_user.email
        )
        db.commit()

        return {
            'raw_division_name': result.raw_division_name,
            'canonical_trade_id': result.canonical_trade_id,
            'canonical_code': result.canonical_code,
            'canonical_name': result.canonical_name,
            'confidence': result.confidence,
            'mapping_method': result.mapping_method,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post(
    "/migrate-project",
    response_model=ProjectMigrationResponse,
    summary="Migrate a project to canonical format",
    description="Run full migration for a single project"
)
def migrate_project(
    request: MigrateProjectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Migrate a single project to the multi-project canonical format."""
    service = ProjectMigrationService(db)

    result = service.migrate_project(
        project_id=request.project_id,
        auto_confirm_threshold=request.auto_confirm_threshold,
        force_remigrate=request.force_remigrate
    )
    db.commit()

    return {
        'project_id': result.project_id,
        'project_code': result.project_code,
        'success': result.success,
        'trades_mapped': result.trades_mapped,
        'trades_auto_confirmed': result.trades_auto_confirmed,
        'trades_need_review': result.trades_need_review,
        'square_feet_inferred': result.square_feet_inferred,
        'data_quality_score': result.data_quality_score,
        'errors': result.errors,
        'warnings': result.warnings,
    }


@router.post(
    "/migrate-all",
    response_model=FullMigrationResponse,
    summary="Migrate all projects",
    description="Run full migration for all projects"
)
def migrate_all_projects(
    auto_confirm_threshold: float = Query(0.9, ge=0.0, le=1.0),
    skip_already_migrated: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Migrate all projects to the multi-project canonical format."""
    service = ProjectMigrationService(db)

    result = service.migrate_all_projects(
        auto_confirm_threshold=auto_confirm_threshold,
        skip_already_migrated=skip_already_migrated
    )

    return {
        'total_projects': result.total_projects,
        'successful': result.successful,
        'failed': result.failed,
        'total_trades_mapped': result.total_trades_mapped,
        'project_results': [
            {
                'project_id': r.project_id,
                'project_code': r.project_code,
                'success': r.success,
                'trades_mapped': r.trades_mapped,
                'trades_auto_confirmed': r.trades_auto_confirmed,
                'trades_need_review': r.trades_need_review,
                'square_feet_inferred': r.square_feet_inferred,
                'data_quality_score': r.data_quality_score,
                'errors': r.errors,
                'warnings': r.warnings,
            }
            for r in result.project_results
        ]
    }


@router.get(
    "/status/{project_id}",
    response_model=MigrationStatusResponse,
    summary="Get migration status",
    description="Get the migration status for a project"
)
def get_migration_status(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get the migration status for a project."""
    service = ProjectMigrationService(db)
    status_data = service.get_migration_status(project_id)

    if 'error' in status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=status_data['error']
        )

    return status_data


@router.post(
    "/backfill-features",
    response_model=List[FeatureBackfillResponse],
    summary="Backfill feature store",
    description="Backfill canonical cost features for ML training"
)
def backfill_features(
    request: BackfillFeaturesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Backfill canonical cost features."""
    service = FeatureStoreService(db)

    if request.project_id:
        # Single project
        result = service.backfill_project_features(
            project_id=request.project_id,
            period_type=request.period_type,
            start_date=request.start_date,
            end_date=request.end_date
        )
        db.commit()

        return [{
            'project_id': result.project_id,
            'periods_created': result.periods_created,
            'trades_covered': result.trades_covered,
            'date_range_start': result.date_range_start.isoformat() if result.date_range_start else None,
            'date_range_end': result.date_range_end.isoformat() if result.date_range_end else None,
            'errors': result.errors,
        }]
    else:
        # All projects
        results = service.backfill_all_projects(period_type=request.period_type)

        return [
            {
                'project_id': r.project_id,
                'periods_created': r.periods_created,
                'trades_covered': r.trades_covered,
                'date_range_start': r.date_range_start.isoformat() if r.date_range_start else None,
                'date_range_end': r.date_range_end.isoformat() if r.date_range_end else None,
                'errors': r.errors,
            }
            for r in results.values()
        ]


@router.get(
    "/unmapped/{project_id}",
    response_model=List[str],
    summary="Get unmapped divisions",
    description="Get GMP divisions without canonical trade mappings"
)
def get_unmapped_divisions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get unmapped GMP divisions for a project."""
    service = TradeMappingService(db)
    return service.get_unmapped_divisions(project_id)


@router.get(
    "/low-confidence/{project_id}",
    summary="Get low confidence mappings",
    description="Get trade mappings below confidence threshold for review"
)
def get_low_confidence_mappings(
    project_id: int,
    threshold: float = Query(0.8, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get low confidence mappings for review."""
    service = TradeMappingService(db)
    mappings = service.get_low_confidence_mappings(project_id, threshold)

    return [
        {
            'id': m.id,
            'raw_division_name': m.raw_division_name,
            'canonical_trade_id': m.canonical_trade_id,
            'confidence': m.confidence,
            'mapping_method': m.mapping_method,
        }
        for m in mappings
    ]
