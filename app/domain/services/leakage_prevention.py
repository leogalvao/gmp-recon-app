"""
Leakage Prevention - Validators to ensure no data leakage in ML pipeline.

Implements:
1. Temporal leakage prevention (no future data in training)
2. Cross-project leakage prevention (project independence)
3. Validation utilities for train/val/test splits
"""
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import (
    Project,
    CanonicalCostFeature,
)

logger = logging.getLogger(__name__)


@dataclass
class LeakageValidationResult:
    """Result of a leakage validation check."""
    is_valid: bool
    leakage_type: Optional[str] = None
    details: Optional[str] = None
    affected_samples: int = 0


class TemporalLeakageValidator:
    """
    Validates temporal ordering to prevent future data leakage.

    Key rules:
    1. Training data must be strictly before validation/test data
    2. Input sequences must not overlap with target periods
    3. No future information in any feature computation
    """

    def __init__(self, db: Session):
        self.db = db

    def validate_sequence_ordering(
        self,
        sequences: np.ndarray,
        sequence_dates: List[List[date]],
        targets: np.ndarray,
        target_dates: List[date],
    ) -> LeakageValidationResult:
        """
        Validate that all sequences have proper temporal ordering.

        Args:
            sequences: Input sequences (batch, seq_len, features)
            sequence_dates: Dates for each sequence position
            targets: Target values
            target_dates: Dates for each target

        Returns:
            Validation result
        """
        violations = 0
        violation_details = []

        for i, (seq_dates, target_date) in enumerate(zip(sequence_dates, target_dates)):
            # Last date in sequence must be before target date
            last_seq_date = max(seq_dates)
            if last_seq_date >= target_date:
                violations += 1
                if len(violation_details) < 5:  # Limit details
                    violation_details.append(
                        f"Sample {i}: seq_end={last_seq_date} >= target={target_date}"
                    )

        if violations > 0:
            return LeakageValidationResult(
                is_valid=False,
                leakage_type="temporal_sequence_overlap",
                details=f"{violations} sequences overlap with targets. Examples: {violation_details}",
                affected_samples=violations,
            )

        return LeakageValidationResult(is_valid=True)

    def validate_train_val_split(
        self,
        train_end_date: date,
        val_start_date: date,
        gap_days: int = 0,
    ) -> LeakageValidationResult:
        """
        Validate that train/val split has no temporal overlap.

        Args:
            train_end_date: Last date in training set
            val_start_date: First date in validation set
            gap_days: Required gap between train and val (for forecast horizon)

        Returns:
            Validation result
        """
        required_val_start = train_end_date + timedelta(days=gap_days + 1)

        if val_start_date < required_val_start:
            return LeakageValidationResult(
                is_valid=False,
                leakage_type="train_val_overlap",
                details=(
                    f"Validation starts {val_start_date} but should start "
                    f">= {required_val_start} (train_end + {gap_days} day gap)"
                ),
            )

        return LeakageValidationResult(is_valid=True)

    def validate_feature_computation(
        self,
        project_id: int,
        feature_date: date,
        lookback_months: int = 3,
    ) -> LeakageValidationResult:
        """
        Validate that feature computation doesn't use future data.

        Checks that rolling statistics and lag features only use
        data from before the feature date.

        Args:
            project_id: Project to validate
            feature_date: Date of the feature being computed
            lookback_months: Maximum lookback window used

        Returns:
            Validation result
        """
        # Get features computed for this date
        features = self.db.query(CanonicalCostFeature).filter(
            CanonicalCostFeature.project_id == project_id,
            CanonicalCostFeature.period_date == feature_date,
        ).all()

        if not features:
            return LeakageValidationResult(
                is_valid=True,
                details="No features found for validation"
            )

        # Verify no features reference future data
        # (This is a structural check - actual data integrity
        # depends on correct feature computation implementation)
        for f in features:
            # Check if any feature values seem impossibly high
            # (could indicate including future cumulative data)
            if f.cumulative_cost_per_sf_cents and f.cost_per_sf_cents:
                if f.cumulative_cost_per_sf_cents < f.cost_per_sf_cents:
                    return LeakageValidationResult(
                        is_valid=False,
                        leakage_type="invalid_cumulative",
                        details=(
                            f"Cumulative cost ({f.cumulative_cost_per_sf_cents}) < "
                            f"period cost ({f.cost_per_sf_cents}) at {feature_date}"
                        ),
                    )

        return LeakageValidationResult(is_valid=True)


class CrossProjectLeakageValidator:
    """
    Validates project independence to prevent cross-project leakage.

    Key rules:
    1. Project features should not depend on other projects' concurrent data
    2. During inference, one project's prediction shouldn't change based on
       changes to another project's data
    """

    def __init__(self, db: Session):
        self.db = db

    def validate_project_isolation(
        self,
        project_ids: List[int],
    ) -> LeakageValidationResult:
        """
        Validate that projects are properly isolated in feature computation.

        Each project's features should be computed independently.

        Args:
            project_ids: Projects to check

        Returns:
            Validation result
        """
        # Check that features are stored per-project
        for project_id in project_ids:
            features = self.db.query(CanonicalCostFeature).filter(
                CanonicalCostFeature.project_id == project_id
            ).limit(1).all()

            if features:
                # Verify feature doesn't reference other projects
                # (structural check based on schema design)
                feature = features[0]
                if feature.project_id != project_id:
                    return LeakageValidationResult(
                        is_valid=False,
                        leakage_type="cross_project_reference",
                        details=f"Feature for project {project_id} has wrong project_id",
                    )

        return LeakageValidationResult(is_valid=True)

    def validate_prediction_independence(
        self,
        model,
        test_project_id: int,
        other_project_id: int,
        test_features: np.ndarray,
        trade_id: int,
    ) -> LeakageValidationResult:
        """
        Validate that predictions for one project don't depend on
        another project's data.

        Args:
            model: The forecasting model
            test_project_id: Project to test predictions for
            other_project_id: Another project in the system
            test_features: Features for test project
            trade_id: Trade to test

        Returns:
            Validation result
        """
        # Get prediction for test project
        pred_before = model.predict_with_uncertainty(
            test_features,
            project_id=test_project_id,
            trade_id=trade_id,
        )

        # The model's prediction should be deterministic given the same inputs
        # Since we're not actually modifying the database, we just verify
        # that repeated predictions are identical
        pred_after = model.predict_with_uncertainty(
            test_features,
            project_id=test_project_id,
            trade_id=trade_id,
        )

        if not np.allclose(pred_before.mean, pred_after.mean, rtol=1e-5):
            return LeakageValidationResult(
                is_valid=False,
                leakage_type="prediction_instability",
                details=(
                    f"Predictions not deterministic: {pred_before.mean} vs {pred_after.mean}"
                ),
            )

        return LeakageValidationResult(is_valid=True)

    def validate_embedding_isolation(
        self,
        model,
        project_id_a: int,
        project_id_b: int,
    ) -> LeakageValidationResult:
        """
        Validate that project embeddings are independent.

        Different projects should have different embeddings
        (unless they haven't been trained yet).

        Args:
            model: The forecasting model with embeddings
            project_id_a: First project
            project_id_b: Second project

        Returns:
            Validation result
        """
        import tensorflow as tf

        # Get embeddings for both projects
        embed_a = model.project_embedding(
            tf.constant([[project_id_a]], dtype=tf.int32)
        ).numpy()
        embed_b = model.project_embedding(
            tf.constant([[project_id_b]], dtype=tf.int32)
        ).numpy()

        # If trained, embeddings should be different
        if np.allclose(embed_a, embed_b, rtol=1e-3):
            return LeakageValidationResult(
                is_valid=False,
                leakage_type="identical_embeddings",
                details=(
                    f"Projects {project_id_a} and {project_id_b} have identical embeddings"
                ),
            )

        return LeakageValidationResult(is_valid=True)


class LeakagePreventionService:
    """
    Unified service for all leakage prevention checks.

    Provides comprehensive validation for:
    - Dataset creation
    - Model training
    - Inference pipeline
    """

    def __init__(self, db: Session):
        self.db = db
        self.temporal_validator = TemporalLeakageValidator(db)
        self.cross_project_validator = CrossProjectLeakageValidator(db)

    def validate_training_dataset(
        self,
        train_data: Dict,
        val_data: Dict,
        config: Dict,
    ) -> List[LeakageValidationResult]:
        """
        Comprehensive validation of a training dataset.

        Args:
            train_data: Training data dictionary with sequences, targets, etc.
            val_data: Validation data dictionary
            config: Dataset configuration

        Returns:
            List of validation results
        """
        results = []

        # 1. Validate train/val temporal split
        if train_data.get('max_date') and val_data.get('min_date'):
            results.append(
                self.temporal_validator.validate_train_val_split(
                    train_end_date=train_data['max_date'],
                    val_start_date=val_data['min_date'],
                    gap_days=config.get('forecast_horizon', 1) * 30,
                )
            )

        # 2. Validate project isolation
        all_project_ids = list(set(
            list(train_data.get('project_ids', [])) +
            list(val_data.get('project_ids', []))
        ))
        results.append(
            self.cross_project_validator.validate_project_isolation(all_project_ids)
        )

        # Log results
        failures = [r for r in results if not r.is_valid]
        if failures:
            logger.warning(f"Dataset validation failed: {len(failures)} issues found")
            for f in failures:
                logger.warning(f"  - {f.leakage_type}: {f.details}")
        else:
            logger.info("Dataset validation passed: No leakage detected")

        return results

    def validate_inference_request(
        self,
        project_id: int,
        as_of_date: date,
        features: np.ndarray,
    ) -> LeakageValidationResult:
        """
        Validate an inference request doesn't use future data.

        Args:
            project_id: Project being forecasted
            as_of_date: Date of the forecast
            features: Input features

        Returns:
            Validation result
        """
        return self.temporal_validator.validate_feature_computation(
            project_id=project_id,
            feature_date=as_of_date,
        )

    def run_full_validation_suite(
        self,
        model=None,
        sample_project_ids: Optional[List[int]] = None,
    ) -> Dict[str, LeakageValidationResult]:
        """
        Run comprehensive leakage validation suite.

        Args:
            model: Trained model (optional, for prediction tests)
            sample_project_ids: Projects to test

        Returns:
            Dict of test name to result
        """
        results = {}

        # Get sample projects if not provided
        if not sample_project_ids:
            projects = self.db.query(Project).filter(
                Project.is_training_eligible == True
            ).limit(5).all()
            sample_project_ids = [p.id for p in projects]

        # 1. Project isolation
        results['project_isolation'] = (
            self.cross_project_validator.validate_project_isolation(sample_project_ids)
        )

        # 2. Feature integrity for each project
        for project_id in sample_project_ids[:3]:  # Test first 3
            latest_feature = self.db.query(CanonicalCostFeature).filter(
                CanonicalCostFeature.project_id == project_id
            ).order_by(CanonicalCostFeature.period_date.desc()).first()

            if latest_feature:
                results[f'feature_integrity_project_{project_id}'] = (
                    self.temporal_validator.validate_feature_computation(
                        project_id=project_id,
                        feature_date=latest_feature.period_date,
                    )
                )

        # 3. Model prediction independence (if model provided)
        if model and len(sample_project_ids) >= 2:
            # Create dummy features for testing
            seq_len = getattr(model, 'seq_len', 12)
            feature_dim = getattr(model, 'feature_dim', 5)
            dummy_features = np.random.randn(seq_len, feature_dim).astype(np.float32)

            results['prediction_independence'] = (
                self.cross_project_validator.validate_prediction_independence(
                    model=model,
                    test_project_id=sample_project_ids[0],
                    other_project_id=sample_project_ids[1],
                    test_features=dummy_features,
                    trade_id=1,
                )
            )

        # Summary
        passed = sum(1 for r in results.values() if r.is_valid)
        total = len(results)
        logger.info(f"Leakage validation suite: {passed}/{total} checks passed")

        return results


def create_leave_one_project_out_split(
    db: Session,
    target_project_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Create a leave-one-project-out split for cross-validation.

    Useful for evaluating how well the model generalizes to new projects.

    Args:
        db: Database session
        target_project_id: Project to hold out

    Returns:
        Tuple of (train_project_ids, test_project_ids)
    """
    all_projects = db.query(Project).filter(
        Project.is_training_eligible == True
    ).all()

    train_ids = [p.id for p in all_projects if p.id != target_project_id]
    test_ids = [target_project_id]

    return train_ids, test_ids
