"""
Project Migration Service - Migrates existing projects to canonical format.

Handles the Phase 2 data migration:
1. Infer project metadata (square footage, type, region)
2. Map GMP divisions to canonical trades
3. Normalize GMP amounts per square foot
4. Compute data quality scores
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import (
    Project,
    GMP,
    BudgetEntity,
    DirectCostEntity,
    CanonicalTrade,
    ProjectTradeMapping,
)
from .trade_mapping_service import TradeMappingService, TradeMappingResult

logger = logging.getLogger(__name__)


@dataclass
class ProjectMigrationResult:
    """Result of migrating a single project."""
    project_id: int
    project_code: str
    success: bool
    trades_mapped: int
    trades_auto_confirmed: int
    trades_need_review: int
    square_feet_inferred: Optional[int]
    data_quality_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FullMigrationResult:
    """Result of migrating all projects."""
    total_projects: int
    successful: int
    failed: int
    total_trades_mapped: int
    project_results: List[ProjectMigrationResult] = field(default_factory=list)


class ProjectMigrationService:
    """
    Service for migrating existing projects to the multi-project platform format.

    Migration Steps:
    1. Infer missing project metadata (square footage, type)
    2. Map all GMP divisions to canonical trades
    3. Normalize GMP amounts per square foot
    4. Compute data quality scores for training eligibility
    """

    # Typical cost per SF by project type (for inference)
    TYPICAL_COST_PER_SF = {
        'commercial': 350_00,  # $350/SF in cents
        'residential': 250_00,
        'industrial': 200_00,
        'healthcare': 500_00,
        'education': 400_00,
        'default': 300_00,
    }

    def __init__(self, db: Session):
        self.db = db
        self.trade_mapping_service = TradeMappingService(db)

    def migrate_project(
        self,
        project_id: int,
        auto_confirm_threshold: float = 0.9,
        force_remigrate: bool = False
    ) -> ProjectMigrationResult:
        """
        Migrate a single project to canonical format.

        Args:
            project_id: Project to migrate
            auto_confirm_threshold: Confidence threshold for auto-confirming mappings
            force_remigrate: Re-run migration even if already done

        Returns:
            Migration result with statistics
        """
        project = self.db.query(Project).get(project_id)
        if not project:
            return ProjectMigrationResult(
                project_id=project_id,
                project_code='UNKNOWN',
                success=False,
                trades_mapped=0,
                trades_auto_confirmed=0,
                trades_need_review=0,
                square_feet_inferred=None,
                data_quality_score=0.0,
                errors=[f"Project {project_id} not found"]
            )

        errors = []
        warnings = []
        sf_inferred = None

        # Step 1: Infer project metadata
        if not project.total_square_feet or force_remigrate:
            inferred_sf = self._infer_square_feet(project)
            if inferred_sf:
                project.total_square_feet = inferred_sf
                sf_inferred = inferred_sf
                logger.info(f"Inferred {inferred_sf} SF for project {project.code}")
            else:
                warnings.append("Could not infer square footage")

        if not project.project_type:
            project.project_type = self._infer_project_type(project)

        # Step 2: Map GMP divisions to canonical trades
        mapping_results = self.trade_mapping_service.map_all_project_divisions(
            project_id=project_id,
            auto_confirm_threshold=auto_confirm_threshold,
            created_by='migration'
        )

        auto_confirmed = sum(1 for r in mapping_results if r.confidence >= auto_confirm_threshold)
        need_review = len(mapping_results) - auto_confirmed

        # Step 3: Normalize GMP amounts per SF
        if project.total_square_feet and project.total_square_feet > 0:
            gmps = self.db.query(GMP).filter(GMP.project_id == project_id).all()
            for gmp in gmps:
                gmp.normalized_amount_per_sf_cents = (
                    gmp.original_amount_cents // project.total_square_feet
                )
        else:
            warnings.append("Cannot normalize amounts - no square footage")

        # Step 4: Compute data quality score
        quality_score = self._compute_data_quality_score(project)
        project.data_quality_score = quality_score

        # Determine training eligibility
        project.is_training_eligible = (
            quality_score >= 0.6 and
            project.total_square_feet is not None and
            project.total_square_feet > 0
        )

        self.db.flush()

        return ProjectMigrationResult(
            project_id=project_id,
            project_code=project.code,
            success=True,
            trades_mapped=len(mapping_results),
            trades_auto_confirmed=auto_confirmed,
            trades_need_review=need_review,
            square_feet_inferred=sf_inferred,
            data_quality_score=quality_score,
            errors=errors,
            warnings=warnings
        )

    def migrate_all_projects(
        self,
        auto_confirm_threshold: float = 0.9,
        skip_already_migrated: bool = True
    ) -> FullMigrationResult:
        """
        Migrate all projects to canonical format.

        Args:
            auto_confirm_threshold: Confidence threshold for auto-confirming
            skip_already_migrated: Skip projects with existing trade mappings

        Returns:
            Full migration result with per-project details
        """
        projects = self.db.query(Project).all()
        results = []
        successful = 0
        failed = 0
        total_trades = 0

        for project in projects:
            # Check if already migrated
            if skip_already_migrated:
                existing_mappings = self.db.query(ProjectTradeMapping).filter(
                    ProjectTradeMapping.project_id == project.id
                ).count()
                if existing_mappings > 0:
                    logger.info(f"Skipping project {project.code} - already migrated")
                    continue

            try:
                result = self.migrate_project(
                    project_id=project.id,
                    auto_confirm_threshold=auto_confirm_threshold
                )
                results.append(result)

                if result.success:
                    successful += 1
                    total_trades += result.trades_mapped
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to migrate project {project.code}: {e}")
                results.append(ProjectMigrationResult(
                    project_id=project.id,
                    project_code=project.code,
                    success=False,
                    trades_mapped=0,
                    trades_auto_confirmed=0,
                    trades_need_review=0,
                    square_feet_inferred=None,
                    data_quality_score=0.0,
                    errors=[str(e)]
                ))
                failed += 1

        self.db.commit()

        return FullMigrationResult(
            total_projects=len(projects),
            successful=successful,
            failed=failed,
            total_trades_mapped=total_trades,
            project_results=results
        )

    def _infer_square_feet(self, project: Project) -> Optional[int]:
        """
        Infer project square footage from total GMP budget.

        Uses typical cost/SF ratios to estimate.
        """
        # Get total GMP amount
        total_gmp = self.db.query(func.sum(GMP.original_amount_cents)).filter(
            GMP.project_id == project.id
        ).scalar() or 0

        if total_gmp == 0:
            return None

        # Use project type cost ratio, or default
        cost_per_sf = self.TYPICAL_COST_PER_SF.get(
            project.project_type,
            self.TYPICAL_COST_PER_SF['default']
        )

        # Calculate estimated SF
        estimated_sf = total_gmp // cost_per_sf

        # Sanity check - SF should be reasonable
        if estimated_sf < 1000 or estimated_sf > 10_000_000:
            return None

        return estimated_sf

    def _infer_project_type(self, project: Project) -> str:
        """
        Infer project type from GMP divisions and descriptions.

        Simple heuristic based on division presence.
        """
        gmps = self.db.query(GMP).filter(GMP.project_id == project.id).all()
        divisions = [gmp.division.lower() for gmp in gmps]
        all_text = ' '.join(divisions)

        # Check for healthcare indicators
        if any(kw in all_text for kw in ['medical', 'hospital', 'clinic', 'healthcare']):
            return 'healthcare'

        # Check for education indicators
        if any(kw in all_text for kw in ['school', 'university', 'education', 'classroom']):
            return 'education'

        # Check for industrial indicators
        if any(kw in all_text for kw in ['warehouse', 'industrial', 'manufacturing']):
            return 'industrial'

        # Check for residential indicators
        if any(kw in all_text for kw in ['residential', 'apartment', 'housing', 'condo']):
            return 'residential'

        # Default to commercial
        return 'commercial'

    def _compute_data_quality_score(self, project: Project) -> float:
        """
        Compute data quality score for training eligibility.

        Factors:
        - Has square footage (0.2)
        - Has complete GMP coverage (0.2)
        - Has budget allocations (0.2)
        - Has direct costs mapped (0.2)
        - Has sufficient history (0.2)
        """
        score = 0.0

        # Factor 1: Has square footage
        if project.total_square_feet and project.total_square_feet > 0:
            score += 0.2

        # Factor 2: GMP coverage
        gmp_count = self.db.query(GMP).filter(GMP.project_id == project.id).count()
        if gmp_count >= 5:  # At least 5 GMP divisions
            score += 0.2
        elif gmp_count > 0:
            score += 0.1

        # Factor 3: Budget allocations
        budget_count = self.db.query(BudgetEntity).join(GMP).filter(
            GMP.project_id == project.id
        ).count()
        if budget_count >= 10:
            score += 0.2
        elif budget_count > 0:
            score += 0.1

        # Factor 4: Direct costs mapped
        mapped_costs = self.db.query(DirectCostEntity).join(BudgetEntity).join(GMP).filter(
            GMP.project_id == project.id
        ).count()
        if mapped_costs >= 50:
            score += 0.2
        elif mapped_costs >= 10:
            score += 0.1

        # Factor 5: Sufficient history (at least 3 months of data)
        if project.start_date:
            from datetime import date
            months_active = (date.today() - project.start_date).days // 30
            if months_active >= 6:
                score += 0.2
            elif months_active >= 3:
                score += 0.1

        return round(score, 2)

    def get_migration_status(self, project_id: int) -> Dict[str, Any]:
        """Get current migration status for a project."""
        project = self.db.query(Project).get(project_id)
        if not project:
            return {'error': 'Project not found'}

        gmp_count = self.db.query(GMP).filter(GMP.project_id == project_id).count()
        mapped_gmps = self.db.query(GMP).filter(
            GMP.project_id == project_id,
            GMP.canonical_trade_id != None
        ).count()

        trade_mappings = self.db.query(ProjectTradeMapping).filter(
            ProjectTradeMapping.project_id == project_id
        ).count()

        low_confidence = self.trade_mapping_service.get_low_confidence_mappings(
            project_id, threshold=0.8
        )

        return {
            'project_id': project_id,
            'project_code': project.code,
            'is_migrated': trade_mappings > 0,
            'total_gmps': gmp_count,
            'gmps_with_canonical_trade': mapped_gmps,
            'trade_mappings_count': trade_mappings,
            'mappings_need_review': len(low_confidence),
            'has_square_feet': project.total_square_feet is not None,
            'square_feet': project.total_square_feet,
            'project_type': project.project_type,
            'data_quality_score': project.data_quality_score,
            'is_training_eligible': project.is_training_eligible,
        }
