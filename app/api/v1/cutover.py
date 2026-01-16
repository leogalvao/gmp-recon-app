"""
Cutover API Endpoints - Project migration from legacy to multi-project system.

Implements Phase 4 integration operations:
- GET /api/v1/cutover/status/{project_id} - Get cutover status
- POST /api/v1/cutover/validate/{project_id} - Validate cutover readiness
- POST /api/v1/cutover/execute/{project_id} - Execute cutover
- POST /api/v1/cutover/rollback/{project_id} - Rollback to legacy
- GET /api/v1/cutover/feature-flags - List feature flag status
- POST /api/v1/cutover/feature-flags/{flag_name} - Update feature flag
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db, User
from app.domain.services import (
    CompatibilityLayer,
    ProjectCutoverService,
    LeakagePreventionService,
)
from app.infrastructure.feature_flags import FeatureFlag, FeatureFlags, RolloutStrategy
from app.api.v1.auth import get_current_active_user

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class CutoverStatusResponse(BaseModel):
    """Response model for cutover status."""
    project_id: int
    project_code: str
    is_multi_project_enabled: bool
    is_ready_for_cutover: bool
    blocking_issues: List[str]
    data_quality_score: Optional[float]


class CutoverValidationResponse(BaseModel):
    """Response model for cutover validation."""
    project_id: int
    is_ready: bool
    issues: List[str]


class CutoverResultResponse(BaseModel):
    """Response model for cutover execution."""
    project_id: int
    project_code: str
    success: bool
    cutover_time: str
    quality_score: float
    forecast_divergence: float
    errors: List[str]
    warnings: List[str]


class FeatureFlagStatus(BaseModel):
    """Status of a feature flag."""
    name: str
    strategy: str
    percentage: float
    allowlist_count: int
    denylist_count: int
    metadata: Dict[str, Any]


class UpdateFeatureFlagRequest(BaseModel):
    """Request to update a feature flag."""
    strategy: Optional[str] = Field(None, description="Strategy: disabled, enabled, percentage, allowlist")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage for rollout")
    entity_id: Optional[int] = Field(None, description="Entity to add/remove from allow/deny list")
    action: Optional[str] = Field(None, description="Action: enable, disable (for entity_id)")


class LeakageValidationResponse(BaseModel):
    """Response model for leakage validation."""
    test_name: str
    is_valid: bool
    leakage_type: Optional[str]
    details: Optional[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/status/{project_id}",
    response_model=CutoverStatusResponse,
    summary="Get cutover status",
    description="Get the current cutover status for a project"
)
def get_cutover_status(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get cutover status for a project."""
    service = ProjectCutoverService(db)
    status = service.get_cutover_status(project_id)

    if 'error' in status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=status['error']
        )

    return status


@router.post(
    "/validate/{project_id}",
    response_model=CutoverValidationResponse,
    summary="Validate cutover readiness",
    description="Validate that a project is ready for cutover"
)
def validate_cutover(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Validate cutover readiness for a project."""
    service = ProjectCutoverService(db)
    is_ready, issues = service.validate_cutover_readiness(project_id)

    return {
        'project_id': project_id,
        'is_ready': is_ready,
        'issues': issues,
    }


@router.post(
    "/execute/{project_id}",
    response_model=CutoverResultResponse,
    summary="Execute cutover",
    description="Execute cutover from legacy to multi-project system"
)
def execute_cutover(
    project_id: int,
    force: bool = Query(False, description="Skip validation checks"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Execute cutover for a project."""
    service = ProjectCutoverService(db)
    result = service.execute_cutover(project_id, force=force)

    return {
        'project_id': result.project_id,
        'project_code': result.project_code,
        'success': result.success,
        'cutover_time': result.cutover_time.isoformat(),
        'quality_score': result.quality_score,
        'forecast_divergence': result.forecast_divergence,
        'errors': result.errors,
        'warnings': result.warnings,
    }


@router.post(
    "/rollback/{project_id}",
    summary="Rollback cutover",
    description="Rollback a project from multi-project to legacy system"
)
def rollback_cutover(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Rollback cutover for a project."""
    service = ProjectCutoverService(db)
    success = service.rollback_cutover(project_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rollback failed"
        )

    return {'message': f'Rollback successful for project {project_id}'}


@router.get(
    "/feature-flags",
    response_model=List[FeatureFlagStatus],
    summary="List feature flags",
    description="List all feature flag statuses"
)
def list_feature_flags(
    current_user: User = Depends(get_current_active_user)
):
    """List all feature flags."""
    return FeatureFlag.list_all()


@router.get(
    "/feature-flags/{flag_name}",
    response_model=FeatureFlagStatus,
    summary="Get feature flag",
    description="Get status of a specific feature flag"
)
def get_feature_flag(
    flag_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific feature flag."""
    flag = FeatureFlag(flag_name)
    return flag.get_status()


@router.post(
    "/feature-flags/{flag_name}",
    response_model=FeatureFlagStatus,
    summary="Update feature flag",
    description="Update a feature flag's configuration"
)
def update_feature_flag(
    flag_name: str,
    request: UpdateFeatureFlagRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Update a feature flag."""
    flag = FeatureFlag(flag_name)

    if request.strategy:
        if request.strategy == 'disabled':
            flag.disable()
        elif request.strategy == 'enabled':
            flag.enable()
        elif request.strategy == 'percentage' and request.percentage is not None:
            flag.set_percentage(request.percentage)

    if request.entity_id is not None and request.action:
        if request.action == 'enable':
            flag.enable(request.entity_id)
        elif request.action == 'disable':
            flag.disable(request.entity_id)

    return flag.get_status()


@router.post(
    "/feature-flags/rollout",
    summary="Set rollout percentage",
    description="Set percentage-based rollout for all multi-project features"
)
def set_rollout_percentage(
    percentage: float = Query(..., ge=0, le=100, description="Rollout percentage"),
    current_user: User = Depends(get_current_active_user)
):
    """Set rollout percentage for all multi-project features."""
    FeatureFlags.set_rollout_percentage(percentage)
    return {'message': f'All features set to {percentage}% rollout'}


@router.get(
    "/leakage-validation",
    response_model=List[LeakageValidationResponse],
    summary="Run leakage validation",
    description="Run leakage prevention validation suite"
)
def run_leakage_validation(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run leakage prevention validation suite."""
    service = LeakagePreventionService(db)
    results = service.run_full_validation_suite()

    return [
        {
            'test_name': name,
            'is_valid': result.is_valid,
            'leakage_type': result.leakage_type,
            'details': result.details,
        }
        for name, result in results.items()
    ]


@router.get(
    "/forecast-comparison/{project_id}",
    summary="Compare forecasts",
    description="Compare legacy and new forecasts for a project"
)
def compare_forecasts(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Compare legacy and new system forecasts."""
    layer = CompatibilityLayer(db)
    comparison = layer.compare_forecasts(project_id)

    return {
        'project_id': comparison.project_id,
        'trade_count': comparison.trade_count,
        'max_divergence': comparison.max_divergence,
        'mean_divergence': comparison.mean_divergence,
        'divergent_trades': comparison.divergent_trades,
    }
