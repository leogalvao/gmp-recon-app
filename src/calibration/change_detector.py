"""
Config Change Detector.

Detects which sections of configuration files have changed,
enabling targeted calibration of only affected components.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Represents a single configuration change."""

    section: str
    old_value: Any
    new_value: Any
    change_type: str  # added, removed, modified
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_type": self.change_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChangeReport:
    """Report of all detected changes."""

    config_file: str
    changes: list[ConfigChange]
    sections_changed: list[str]
    calibration_targets: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    def to_dict(self) -> dict:
        return {
            "config_file": self.config_file,
            "changes": [c.to_dict() for c in self.changes],
            "sections_changed": self.sections_changed,
            "calibration_targets": self.calibration_targets,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfigChangeDetector:
    """
    Detects changes in configuration files by comparing against stored snapshots.

    Usage:
        detector = ConfigChangeDetector()
        report = detector.detect_changes("gmp_mapping_config.yaml")
        if report.has_changes:
            print(f"Sections changed: {report.sections_changed}")
            print(f"Calibration needed: {report.calibration_targets}")
    """

    def __init__(
        self,
        config_dir: str = ".",
        snapshot_dir: str = ".config_snapshots",
        registry_path: str = "config/calibration_registry.yaml",
    ):
        self.config_dir = Path(config_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.registry_path = Path(registry_path)
        self._registry = None

    @property
    def registry(self) -> dict:
        """Load calibration registry on first access."""
        if self._registry is None:
            if self.registry_path.exists():
                with open(self.registry_path) as f:
                    self._registry = yaml.safe_load(f)
            else:
                self._registry = {"config_mappings": {}, "dependencies": {}}
        return self._registry

    def _get_snapshot_path(self, config_file: str) -> Path:
        """Get path to snapshot file for a config."""
        safe_name = config_file.replace("/", "_").replace(".", "_")
        return self.snapshot_dir / f"{safe_name}.snapshot.json"

    def _load_snapshot(self, config_file: str) -> dict | None:
        """Load previous config snapshot."""
        snapshot_path = self._get_snapshot_path(config_file)
        if snapshot_path.exists():
            with open(snapshot_path) as f:
                return json.load(f)
        return None

    def _save_snapshot(self, config_file: str, config_data: dict) -> None:
        """Save current config as snapshot."""
        snapshot_path = self._get_snapshot_path(config_file)
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "config_file": config_file,
            "data": config_data,
            "hash": self._hash_config(config_data),
        }
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    def _hash_config(self, config_data: dict) -> str:
        """Generate hash of config for quick comparison."""
        content = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _flatten_dict(
        self, d: dict, parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten nested dict with dot-notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _compare_configs(
        self, old_config: dict, new_config: dict
    ) -> list[ConfigChange]:
        """Compare two configs and return list of changes."""
        changes = []

        old_flat = self._flatten_dict(old_config)
        new_flat = self._flatten_dict(new_config)

        all_keys = set(old_flat.keys()) | set(new_flat.keys())

        for key in all_keys:
            old_val = old_flat.get(key)
            new_val = new_flat.get(key)

            if old_val is None and new_val is not None:
                changes.append(
                    ConfigChange(
                        section=key,
                        old_value=None,
                        new_value=new_val,
                        change_type="added",
                    )
                )
            elif old_val is not None and new_val is None:
                changes.append(
                    ConfigChange(
                        section=key,
                        old_value=old_val,
                        new_value=None,
                        change_type="removed",
                    )
                )
            elif old_val != new_val:
                changes.append(
                    ConfigChange(
                        section=key,
                        old_value=old_val,
                        new_value=new_val,
                        change_type="modified",
                    )
                )

        return changes

    def _get_top_level_sections(self, changes: list[ConfigChange]) -> list[str]:
        """Extract top-level section names from changes."""
        sections = set()
        for change in changes:
            # Get first level of dot-notation path
            top_section = change.section.split(".")[0]
            sections.add(top_section)
        return sorted(sections)

    def _get_calibration_targets(
        self, config_name: str, changes: list[ConfigChange]
    ) -> list[str]:
        """Determine which calibration targets are affected by changes."""
        targets = set()
        config_mappings = self.registry.get("config_mappings", {}).get(
            config_name.replace(".yaml", "").replace("-", "_"), {}
        )

        for change in changes:
            section = change.section
            # Check exact match first
            if section in config_mappings:
                for target in config_mappings[section].get("targets", []):
                    targets.add(target)
            else:
                # Check prefix matches
                for mapped_section, mapping in config_mappings.items():
                    if section.startswith(mapped_section):
                        for target in mapping.get("targets", []):
                            targets.add(target)

        # Add dependency targets
        dependencies = self.registry.get("dependencies", {})
        for target in list(targets):
            if target in dependencies:
                for triggered in dependencies[target].get("triggers", []):
                    targets.add(triggered)

        return sorted(targets)

    def detect_changes(
        self, config_file: str, update_snapshot: bool = False
    ) -> ChangeReport:
        """
        Detect changes in a configuration file.

        Args:
            config_file: Name of config file (e.g., "gmp_mapping_config.yaml")
            update_snapshot: If True, update snapshot after detection

        Returns:
            ChangeReport with detected changes and affected calibration targets
        """
        config_path = self.config_dir / config_file
        if not config_path.exists():
            # Try config subdirectory
            config_path = self.config_dir / "config" / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Load current config
        with open(config_path) as f:
            current_config = yaml.safe_load(f)

        # Load previous snapshot
        snapshot = self._load_snapshot(config_file)

        if snapshot is None:
            # No previous snapshot - treat as all new
            logger.info(f"No previous snapshot for {config_file}, creating baseline")
            if update_snapshot:
                self._save_snapshot(config_file, current_config)
            return ChangeReport(
                config_file=config_file,
                changes=[],
                sections_changed=[],
                calibration_targets=[],
            )

        old_config = snapshot.get("data", {})

        # Quick hash check
        old_hash = snapshot.get("hash")
        new_hash = self._hash_config(current_config)

        if old_hash == new_hash:
            logger.info(f"No changes detected in {config_file}")
            return ChangeReport(
                config_file=config_file,
                changes=[],
                sections_changed=[],
                calibration_targets=[],
            )

        # Detailed comparison
        changes = self._compare_configs(old_config, current_config)
        sections_changed = self._get_top_level_sections(changes)
        calibration_targets = self._get_calibration_targets(config_file, changes)

        logger.info(
            f"Detected {len(changes)} changes in {config_file} "
            f"affecting sections: {sections_changed}"
        )

        if update_snapshot:
            self._save_snapshot(config_file, current_config)

        return ChangeReport(
            config_file=config_file,
            changes=changes,
            sections_changed=sections_changed,
            calibration_targets=calibration_targets,
        )

    def detect_all_changes(
        self, update_snapshot: bool = False
    ) -> dict[str, ChangeReport]:
        """
        Detect changes in all known configuration files.

        Returns:
            Dict mapping config file names to their change reports
        """
        config_files = ["gmp_mapping_config.yaml", "config/training_config.yaml"]
        reports = {}

        for config_file in config_files:
            try:
                reports[config_file] = self.detect_changes(
                    config_file, update_snapshot=update_snapshot
                )
            except FileNotFoundError:
                logger.warning(f"Config file not found: {config_file}")

        return reports

    def create_baseline(self, config_file: str | None = None) -> None:
        """
        Create baseline snapshots for config files.

        Args:
            config_file: Specific file to snapshot, or None for all
        """
        if config_file:
            config_files = [config_file]
        else:
            config_files = ["gmp_mapping_config.yaml", "config/training_config.yaml"]

        for cf in config_files:
            try:
                config_path = self.config_dir / cf
                if not config_path.exists():
                    config_path = self.config_dir / "config" / cf

                if config_path.exists():
                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)
                    self._save_snapshot(cf, config_data)
                    logger.info(f"Created baseline snapshot for {cf}")
            except Exception as e:
                logger.error(f"Failed to create baseline for {cf}: {e}")

    def get_snapshot_info(self, config_file: str) -> dict | None:
        """Get information about a config's snapshot."""
        snapshot = self._load_snapshot(config_file)
        if snapshot:
            return {
                "config_file": snapshot.get("config_file"),
                "timestamp": snapshot.get("timestamp"),
                "hash": snapshot.get("hash"),
            }
        return None
