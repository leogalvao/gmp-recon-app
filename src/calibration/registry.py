"""
Calibration Registry.

Provides access to calibration target definitions, dependencies,
and profile configurations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


CalibrationType = Literal[
    "parameter_update",
    "scoring_recalc",
    "lookup_rebuild",
    "matrix_update",
    "model_retrain",
    "feature_engineering",
    "mapping_rebuild",
]


@dataclass
class CalibrationTarget:
    """Definition of a calibration target."""

    name: str
    description: str
    calibration_type: CalibrationType
    estimated_time_seconds: int
    requires_data: bool = False
    data_sources: list[str] = field(default_factory=list)
    affects: list[str] = field(default_factory=list)

    @property
    def is_lightweight(self) -> bool:
        """Check if this is a lightweight calibration (no model training)."""
        return self.calibration_type in [
            "parameter_update",
            "lookup_rebuild",
            "matrix_update",
        ]


@dataclass
class CalibrationProfile:
    """Predefined calibration profile."""

    name: str
    description: str
    targets: list[str] | Literal["all"]
    include_types: list[CalibrationType] | None = None
    exclude_types: list[CalibrationType] | None = None
    estimated_time_minutes: int | None = None


class CalibrationRegistry:
    """
    Registry for calibration targets and profiles.

    Usage:
        registry = CalibrationRegistry()
        target = registry.get_target("fuzzy_matching")
        targets = registry.get_profile_targets("quick")
    """

    def __init__(self, registry_path: str = "config/calibration_registry.yaml"):
        self.registry_path = Path(registry_path)
        self._data = None
        self._targets: dict[str, CalibrationTarget] = {}
        self._profiles: dict[str, CalibrationProfile] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from YAML file."""
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            self._data = {}
            return

        with open(self.registry_path) as f:
            self._data = yaml.safe_load(f)

        # Parse targets
        for name, config in self._data.get("targets", {}).items():
            self._targets[name] = CalibrationTarget(
                name=name,
                description=config.get("description", ""),
                calibration_type=config.get("calibration_type", "parameter_update"),
                estimated_time_seconds=config.get("estimated_time_seconds", 60),
                requires_data=config.get("requires_data", False),
                data_sources=config.get("data_sources", []),
                affects=config.get("affects", []),
            )

        # Parse profiles
        for name, config in self._data.get("profiles", {}).items():
            self._profiles[name] = CalibrationProfile(
                name=name,
                description=config.get("description", ""),
                targets=config.get("targets", []),
                include_types=config.get("include_types"),
                exclude_types=config.get("exclude_types"),
                estimated_time_minutes=config.get("estimated_time_minutes"),
            )

    def get_target(self, name: str) -> CalibrationTarget | None:
        """Get a specific calibration target by name."""
        return self._targets.get(name)

    def get_all_targets(self) -> dict[str, CalibrationTarget]:
        """Get all calibration targets."""
        return self._targets.copy()

    def get_profile(self, name: str) -> CalibrationProfile | None:
        """Get a specific calibration profile by name."""
        return self._profiles.get(name)

    def get_all_profiles(self) -> dict[str, CalibrationProfile]:
        """Get all calibration profiles."""
        return self._profiles.copy()

    def get_profile_targets(self, profile_name: str) -> list[CalibrationTarget]:
        """
        Get all targets for a profile, resolving includes/excludes.

        Args:
            profile_name: Name of the profile

        Returns:
            List of CalibrationTarget objects for the profile
        """
        profile = self._profiles.get(profile_name)
        if not profile:
            return []

        if profile.targets == "all":
            targets = list(self._targets.values())
        else:
            targets = [
                self._targets[name]
                for name in profile.targets
                if name in self._targets
            ]

        # Filter by include types
        if profile.include_types:
            targets = [t for t in targets if t.calibration_type in profile.include_types]

        # Filter by exclude types
        if profile.exclude_types:
            targets = [
                t for t in targets if t.calibration_type not in profile.exclude_types
            ]

        return targets

    def get_dependencies(self, target_name: str) -> list[str]:
        """Get targets that should be triggered when target_name is calibrated."""
        dependencies = self._data.get("dependencies", {})
        return dependencies.get(target_name, {}).get("triggers", [])

    def resolve_dependencies(self, target_names: list[str]) -> list[str]:
        """
        Resolve all dependencies for a list of targets.

        Returns target names in order they should be calibrated.
        """
        all_targets = set(target_names)
        to_process = list(target_names)

        while to_process:
            current = to_process.pop(0)
            for triggered in self.get_dependencies(current):
                if triggered not in all_targets:
                    all_targets.add(triggered)
                    to_process.append(triggered)

        # Sort by priority (lightweight first, then by estimated time)
        def sort_key(name: str) -> tuple:
            target = self._targets.get(name)
            if not target:
                return (1, 1000)
            return (0 if target.is_lightweight else 1, target.estimated_time_seconds)

        return sorted(all_targets, key=sort_key)

    def get_targets_for_sections(
        self, config_name: str, sections: list[str]
    ) -> list[str]:
        """
        Get calibration targets affected by config sections.

        Args:
            config_name: Name of config file (without extension)
            sections: List of section names that changed

        Returns:
            List of target names that need calibration
        """
        config_mappings = self._data.get("config_mappings", {}).get(config_name, {})
        targets = set()

        for section in sections:
            # Check exact match
            if section in config_mappings:
                targets.update(config_mappings[section].get("targets", []))
            else:
                # Check prefix matches
                for mapped_section, mapping in config_mappings.items():
                    if section.startswith(mapped_section + "."):
                        targets.update(mapping.get("targets", []))

        return list(targets)

    def estimate_time(self, target_names: list[str]) -> int:
        """
        Estimate total time in seconds for calibrating targets.

        Args:
            target_names: List of target names to calibrate

        Returns:
            Estimated total time in seconds
        """
        total = 0
        for name in target_names:
            target = self._targets.get(name)
            if target:
                total += target.estimated_time_seconds
        return total

    def get_lightweight_targets(self) -> list[CalibrationTarget]:
        """Get all lightweight (no model training) targets."""
        return [t for t in self._targets.values() if t.is_lightweight]

    def get_data_requirements(self, target_names: list[str]) -> set[str]:
        """Get all data sources required for targets."""
        sources = set()
        for name in target_names:
            target = self._targets.get(name)
            if target:
                sources.update(target.data_sources)
        return sources
