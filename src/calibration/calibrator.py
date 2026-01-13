"""
Modular Calibrator.

Orchestrates targeted calibration of specific system components
based on configuration changes.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml

from .change_detector import ChangeReport, ConfigChangeDetector
from .registry import CalibrationRegistry, CalibrationTarget

logger = logging.getLogger(__name__)


class CalibrationStatus(Enum):
    """Status of a calibration run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CalibrationResult:
    """Result of a single calibration target."""

    target_name: str
    status: CalibrationStatus
    duration_seconds: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CalibrationSummary:
    """Summary of a complete calibration run."""

    targets_run: list[str]
    results: list[CalibrationResult]
    total_duration_seconds: float
    status: CalibrationStatus
    profile_used: str | None = None
    triggered_by: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def successful_count(self) -> int:
        return sum(1 for r in self.results if r.status == CalibrationStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == CalibrationStatus.FAILED)

    def to_dict(self) -> dict:
        return {
            "targets_run": self.targets_run,
            "results": [r.to_dict() for r in self.results],
            "total_duration_seconds": self.total_duration_seconds,
            "status": self.status.value,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "profile_used": self.profile_used,
            "triggered_by": self.triggered_by,
            "timestamp": self.timestamp.isoformat(),
        }


class Calibrator:
    """
    Main calibration orchestrator.

    Handles targeted calibration of system components based on
    configuration changes or explicit requests.

    Usage:
        calibrator = Calibrator()

        # Calibrate specific targets
        summary = calibrator.calibrate(["fuzzy_matching", "suggestion_engine"])

        # Use a profile
        summary = calibrator.calibrate_profile("quick")

        # Auto-detect changes and calibrate
        summary = calibrator.auto_calibrate()
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        model_dir: str = "models",
        config_dir: str = ".",
        registry_path: str = "config/calibration_registry.yaml",
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.config_dir = Path(config_dir)
        self.registry = CalibrationRegistry(registry_path)
        self.change_detector = ConfigChangeDetector(
            config_dir=str(config_dir),
            registry_path=registry_path,
        )

        # Registry of calibration handlers
        self._handlers: dict[str, Callable[[CalibrationTarget], CalibrationResult]] = {
            "fuzzy_matching": self._calibrate_fuzzy_matching,
            "suggestion_engine": self._calibrate_suggestion_engine,
            "gmp_divisions": self._calibrate_gmp_divisions,
            "allocations": self._calibrate_allocations,
            "duplicate_detection": self._calibrate_duplicate_detection,
            "ml_forecasting": self._calibrate_ml_forecasting,
            "forecasting_methods": self._calibrate_forecasting_methods,
            "schedule_parsing": self._calibrate_schedule_parsing,
            "schedule_models": self._calibrate_schedule_models,
            "cost_type_compatibility": self._calibrate_cost_type_compatibility,
            "building_features": self._calibrate_building_features,
            "training_hyperparams": self._calibrate_training_hyperparams,
        }

    def calibrate(
        self,
        target_names: list[str],
        resolve_dependencies: bool = True,
        dry_run: bool = False,
    ) -> CalibrationSummary:
        """
        Calibrate specific targets.

        Args:
            target_names: List of target names to calibrate
            resolve_dependencies: If True, also calibrate dependent targets
            dry_run: If True, only show what would be done

        Returns:
            CalibrationSummary with results
        """
        start_time = time.time()
        results = []

        # Resolve dependencies
        if resolve_dependencies:
            target_names = self.registry.resolve_dependencies(target_names)

        logger.info(f"Calibrating targets: {target_names}")

        if dry_run:
            for name in target_names:
                target = self.registry.get_target(name)
                if target:
                    results.append(
                        CalibrationResult(
                            target_name=name,
                            status=CalibrationStatus.SKIPPED,
                            duration_seconds=0,
                            message=f"DRY RUN: Would calibrate {target.description}",
                            details={
                                "calibration_type": target.calibration_type,
                                "estimated_time": target.estimated_time_seconds,
                            },
                        )
                    )
        else:
            for name in target_names:
                result = self._run_calibration(name)
                results.append(result)

        total_duration = time.time() - start_time

        # Determine overall status
        if any(r.status == CalibrationStatus.FAILED for r in results):
            overall_status = CalibrationStatus.FAILED
        elif all(r.status == CalibrationStatus.SKIPPED for r in results):
            overall_status = CalibrationStatus.SKIPPED
        else:
            overall_status = CalibrationStatus.COMPLETED

        return CalibrationSummary(
            targets_run=target_names,
            results=results,
            total_duration_seconds=total_duration,
            status=overall_status,
        )

    def calibrate_profile(
        self, profile_name: str, dry_run: bool = False
    ) -> CalibrationSummary:
        """
        Calibrate using a predefined profile.

        Args:
            profile_name: Name of the profile (quick, mapping, forecasting, etc.)
            dry_run: If True, only show what would be done

        Returns:
            CalibrationSummary with results
        """
        targets = self.registry.get_profile_targets(profile_name)
        if not targets:
            return CalibrationSummary(
                targets_run=[],
                results=[],
                total_duration_seconds=0,
                status=CalibrationStatus.SKIPPED,
                profile_used=profile_name,
            )

        target_names = [t.name for t in targets]
        summary = self.calibrate(target_names, resolve_dependencies=False, dry_run=dry_run)
        summary.profile_used = profile_name
        return summary

    def auto_calibrate(
        self,
        update_snapshots: bool = True,
        dry_run: bool = False,
    ) -> CalibrationSummary:
        """
        Auto-detect config changes and calibrate only affected targets.

        Args:
            update_snapshots: If True, update config snapshots after calibration
            dry_run: If True, only show what would be done

        Returns:
            CalibrationSummary with results
        """
        # Detect changes in all config files
        all_targets = set()
        triggered_by = []

        reports = self.change_detector.detect_all_changes(
            update_snapshot=update_snapshots
        )

        for config_file, report in reports.items():
            if report.has_changes:
                triggered_by.append(config_file)
                all_targets.update(report.calibration_targets)

        if not all_targets:
            logger.info("No configuration changes detected, nothing to calibrate")
            return CalibrationSummary(
                targets_run=[],
                results=[],
                total_duration_seconds=0,
                status=CalibrationStatus.SKIPPED,
                triggered_by="No changes detected",
            )

        summary = self.calibrate(list(all_targets), dry_run=dry_run)
        summary.triggered_by = ", ".join(triggered_by)
        return summary

    def _run_calibration(self, target_name: str) -> CalibrationResult:
        """Run calibration for a single target."""
        target = self.registry.get_target(target_name)
        if not target:
            return CalibrationResult(
                target_name=target_name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Unknown calibration target: {target_name}",
            )

        handler = self._handlers.get(target_name)
        if not handler:
            return CalibrationResult(
                target_name=target_name,
                status=CalibrationStatus.SKIPPED,
                duration_seconds=0,
                message=f"No handler registered for target: {target_name}",
            )

        logger.info(f"Running calibration: {target_name} ({target.calibration_type})")
        start_time = time.time()

        try:
            result = handler(target)
            result.duration_seconds = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"Calibration failed for {target_name}: {e}")
            return CalibrationResult(
                target_name=target_name,
                status=CalibrationStatus.FAILED,
                duration_seconds=time.time() - start_time,
                message=str(e),
            )

    # =========================================================================
    # CALIBRATION HANDLERS
    # =========================================================================
    # Each handler implements calibration logic for a specific target.
    # Handlers can be lightweight (parameter updates) or heavy (model retraining).

    def _calibrate_fuzzy_matching(self, target: CalibrationTarget) -> CalibrationResult:
        """Calibrate fuzzy matching parameters."""
        try:
            # Reload config to get updated thresholds
            from app.config import reload_config

            config = reload_config()
            fuzzy_config = config.get("fuzzy_matching", {})

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Fuzzy matching parameters reloaded",
                details={
                    "algorithm": fuzzy_config.get("algorithm"),
                    "thresholds": fuzzy_config.get("thresholds", {}),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to reload fuzzy matching config: {e}",
            )

    def _calibrate_suggestion_engine(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Recalibrate suggestion engine scoring."""
        try:
            from app.config import reload_config

            config = reload_config()
            suggestion_config = config.get("suggestion_engine", {})

            # Rebuild scoring weights
            weights = suggestion_config.get("weights", {})

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Suggestion engine weights recalibrated",
                details={
                    "weights": weights,
                    "confidence_bands": suggestion_config.get("confidence_bands", {}),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to recalibrate suggestion engine: {e}",
            )

    def _calibrate_gmp_divisions(self, target: CalibrationTarget) -> CalibrationResult:
        """Rebuild GMP division lookup cache."""
        try:
            from app.config import reload_config

            config = reload_config()
            divisions = config.get("gmp_divisions", {})

            # Build lookup cache with aliases
            lookup_cache = {}
            for key, div in divisions.items():
                lookup_cache[key] = div.get("name")
                for alias in div.get("aliases", []):
                    lookup_cache[alias.lower()] = div.get("name")

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message=f"GMP division cache rebuilt with {len(divisions)} divisions",
                details={
                    "division_count": len(divisions),
                    "alias_count": len(lookup_cache) - len(divisions),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to rebuild GMP divisions: {e}",
            )

    def _calibrate_allocations(self, target: CalibrationTarget) -> CalibrationResult:
        """Update regional allocation defaults."""
        try:
            from app.config import reload_config

            config = reload_config()
            allocations = config.get("allocations", {})

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Allocation defaults updated",
                details={
                    "default_split": allocations.get("default", {}),
                    "division_overrides": len(
                        allocations.get("division_defaults", {})
                    ),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to update allocations: {e}",
            )

    def _calibrate_duplicate_detection(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Update duplicate detection thresholds."""
        try:
            from app.config import reload_config

            config = reload_config()
            dup_config = config.get("duplicate_detection", {})

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Duplicate detection thresholds updated",
                details={
                    "methods_enabled": [
                        m
                        for m, cfg in dup_config.get("methods", {}).items()
                        if cfg.get("enabled", False)
                    ],
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to update duplicate detection: {e}",
            )

    def _calibrate_ml_forecasting(self, target: CalibrationTarget) -> CalibrationResult:
        """Retrain ML forecasting models."""
        try:
            import pandas as pd

            from app.infrastructure.ml.training_pipeline import (
                TrainingConfig,
                TrainingPipeline,
            )

            # Load training config
            config_path = self.config_dir / "config" / "training_config.yaml"
            if not config_path.exists():
                config_path = self.config_dir / "training_config.yaml"

            pipeline_config = TrainingConfig.from_yaml(str(config_path))
            pipeline = TrainingPipeline(config=pipeline_config)

            # Check for data files
            costs_path = self.data_dir / "historical_costs.csv"
            buildings_path = self.data_dir / "buildings.csv"

            if not costs_path.exists() or not buildings_path.exists():
                return CalibrationResult(
                    target_name=target.name,
                    status=CalibrationStatus.SKIPPED,
                    duration_seconds=0,
                    message="Training data not found, skipping model retrain",
                    details={
                        "costs_exists": costs_path.exists(),
                        "buildings_exists": buildings_path.exists(),
                    },
                )

            # Load data and train
            historical_costs = pd.read_csv(costs_path)
            building_df = pd.read_csv(buildings_path)

            history = pipeline.train(historical_costs, building_df)

            # Save model
            output_path = self.model_dir / "model.keras"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline.save(str(output_path))

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="ML forecasting model retrained",
                details={
                    "final_loss": history["loss"][-1] if history.get("loss") else None,
                    "epochs_trained": len(history.get("loss", [])),
                    "model_path": str(output_path),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to retrain ML model: {e}",
            )

    def _calibrate_forecasting_methods(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Update forecasting method parameters."""
        try:
            from app.config import reload_config

            config = reload_config()
            forecast_config = config.get("forecasting", {})

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Forecasting method parameters updated",
                details={
                    "default_method": forecast_config.get("default_method"),
                    "evm_config": forecast_config.get("evm", {}),
                    "pert_config": forecast_config.get("pert", {}),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to update forecasting methods: {e}",
            )

    def _calibrate_schedule_parsing(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Rebuild schedule activity-trade mappings."""
        try:
            import pandas as pd

            from src.schedule.parser import ScheduleParser

            schedule_path = self.data_dir / "schedule.csv"
            if not schedule_path.exists():
                return CalibrationResult(
                    target_name=target.name,
                    status=CalibrationStatus.SKIPPED,
                    duration_seconds=0,
                    message="schedule.csv not found",
                )

            schedule_df = pd.read_csv(schedule_path)
            parser = ScheduleParser(schedule_df)

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Schedule mappings rebuilt",
                details={
                    "activities_parsed": len(parser.activities),
                    "phases_parsed": len(parser.phases),
                    "trades_mapped": len(parser.get_trade_summary()),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to rebuild schedule mappings: {e}",
            )

    def _calibrate_schedule_models(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Retrain schedule-driven forecasting models."""
        try:
            import pandas as pd

            from src.data.loaders import DataLoader
            from src.training.schedule_driven_trainer import ScheduleDrivenTrainer

            loader = DataLoader(str(self.data_dir))
            data = loader.load_all()

            required = ["schedule", "breakdown", "direct_costs"]
            missing = [k for k in required if data.get(k) is None]
            if missing:
                return CalibrationResult(
                    target_name=target.name,
                    status=CalibrationStatus.SKIPPED,
                    duration_seconds=0,
                    message=f"Missing data files: {missing}",
                )

            trainer = ScheduleDrivenTrainer()
            trainer.prepare(
                schedule_df=data["schedule"],
                gmp_breakdown_df=data["breakdown"],
                direct_costs_df=data["direct_costs"],
                budget_df=data.get("budget"),
            )

            results = trainer.train(epochs=50)  # Quick retrain

            # Save models
            output_dir = self.model_dir / "schedule_driven"
            trainer.save(str(output_dir))

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Schedule-driven models retrained",
                details={
                    "models_trained": len(results),
                    "output_dir": str(output_dir),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to retrain schedule models: {e}",
            )

    def _calibrate_cost_type_compatibility(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Update cost type compatibility matrix."""
        try:
            from app.config import reload_config

            config = reload_config()
            suggestion_config = config.get("suggestion_engine", {})
            compatibility = suggestion_config.get("cost_type_compatibility", [])

            # Build matrix
            matrix = {}
            for entry in compatibility:
                types = tuple(entry.get("types", []))
                matrix[types] = entry.get("score", 0.2)

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Cost type compatibility matrix updated",
                details={
                    "matrix_entries": len(matrix),
                    "default_score": suggestion_config.get("default_score", 0.2),
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to update compatibility matrix: {e}",
            )

    def _calibrate_building_features(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Recalibrate building feature normalization."""
        try:
            import pandas as pd

            from app.infrastructure.ml.feature_engineering import FeatureEngineer

            buildings_path = self.data_dir / "buildings.csv"
            if not buildings_path.exists():
                return CalibrationResult(
                    target_name=target.name,
                    status=CalibrationStatus.SKIPPED,
                    duration_seconds=0,
                    message="buildings.csv not found",
                )

            building_df = pd.read_csv(buildings_path)
            engineer = FeatureEngineer()
            engineer.fit(building_df)

            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.COMPLETED,
                duration_seconds=0,
                message="Building feature normalization recalibrated",
                details={
                    "buildings_processed": len(building_df),
                    "features_fitted": True,
                },
            )
        except Exception as e:
            return CalibrationResult(
                target_name=target.name,
                status=CalibrationStatus.FAILED,
                duration_seconds=0,
                message=f"Failed to recalibrate features: {e}",
            )

    def _calibrate_training_hyperparams(
        self, target: CalibrationTarget
    ) -> CalibrationResult:
        """Apply new training hyperparameters (triggers full retrain)."""
        # This delegates to ml_forecasting since changing hyperparams requires retrain
        return self._calibrate_ml_forecasting(target)
