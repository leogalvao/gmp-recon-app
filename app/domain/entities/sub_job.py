"""
Sub-Job Entity - Temporal decomposition of project work.

Models sub-jobs that:
- Start sequentially (east-first pattern)
- Have overlapping design phases
- Share ~1 month construction overlap
"""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4


class SubJobPhase(Enum):
    """Project phase status for a sub-job."""
    NOT_STARTED = "not_started"
    DESIGN = "design"
    PRECONSTRUCTION = "preconstruction"
    CONSTRUCTION = "construction"
    CLOSEOUT = "closeout"
    COMPLETE = "complete"


@dataclass
class PhaseTimeline:
    """
    Timeline for a specific project phase.

    Tracks both planned and actual dates for a single phase.

    Attributes:
        phase: The phase this timeline represents
        planned_start: Planned start date
        planned_end: Planned end date
        actual_start: Actual start date
        actual_end: Actual end date
    """

    phase: SubJobPhase
    planned_start: Optional[date] = None
    planned_end: Optional[date] = None
    actual_start: Optional[date] = None
    actual_end: Optional[date] = None

    def duration_days(self) -> int:
        """Planned duration in days."""
        if self.planned_start and self.planned_end:
            return (self.planned_end - self.planned_start).days
        return 0

    def actual_duration_days(self) -> Optional[int]:
        """Actual duration in days (if completed)."""
        if self.actual_start and self.actual_end:
            return (self.actual_end - self.actual_start).days
        return None

    def is_overlapping(self, other: 'PhaseTimeline') -> bool:
        """
        Check if this phase overlaps with another.

        Args:
            other: Another PhaseTimeline to compare

        Returns:
            True if phases overlap in time
        """
        if not all([self.planned_start, self.planned_end,
                   other.planned_start, other.planned_end]):
            return False
        return (self.planned_start <= other.planned_end and
                self.planned_end >= other.planned_start)

    def overlap_days(self, other: 'PhaseTimeline') -> int:
        """
        Calculate overlap duration in days.

        Args:
            other: Another PhaseTimeline to compare

        Returns:
            Number of overlapping days (0 if no overlap)
        """
        if not self.is_overlapping(other):
            return 0
        overlap_start = max(self.planned_start, other.planned_start)
        overlap_end = min(self.planned_end, other.planned_end)
        return (overlap_end - overlap_start).days

    def schedule_variance_days(self) -> Optional[int]:
        """
        Calculate schedule variance.

        Returns:
            Positive = ahead of schedule
            Negative = behind schedule
            None = phase not complete
        """
        if not self.actual_end or not self.planned_end:
            return None
        return (self.planned_end - self.actual_end).days


@dataclass
class BuildingParameters:
    """
    Physical parameters affecting cost forecasting.

    These parameters are used as features for ML-based
    cost prediction models.

    Attributes:
        square_feet: Total building square footage
        stories: Number of floors
        has_green_roof: Whether building has a green roof
        rooftop_units_qty: Number of rooftop HVAC units
        fall_anchor_count: Number of fall protection anchors
        building_type: Type of building (commercial, residential, industrial)
        foundation_type: Foundation system (slab, basement, crawl)
        structural_system: Primary structural system (steel, concrete, wood)
        facade_type: Facade system (curtain_wall, masonry, precast)
    """

    square_feet: float = 0.0
    stories: int = 1
    has_green_roof: bool = False
    rooftop_units_qty: int = 0
    fall_anchor_count: int = 0

    # Additional parameters for forecasting
    building_type: str = "commercial"  # commercial, residential, industrial
    foundation_type: str = "slab"  # slab, basement, crawl
    structural_system: str = "steel"  # steel, concrete, wood
    facade_type: str = "curtain_wall"  # curtain_wall, masonry, precast

    def to_feature_vector(self) -> List[float]:
        """
        Convert to ML feature vector.

        Returns:
            List of numeric features for model input
        """
        return [
            self.square_feet,
            float(self.stories),
            1.0 if self.has_green_roof else 0.0,
            float(self.rooftop_units_qty),
            float(self.fall_anchor_count),
        ]

    def sqft_per_story(self) -> float:
        """Calculate average square footage per story."""
        if self.stories == 0:
            return 0.0
        return self.square_feet / self.stories

    def complexity_score(self) -> float:
        """
        Calculate building complexity score (0-1).

        Considers green roof, rooftop units, and fall anchors
        as indicators of construction complexity.
        """
        score = (
            (1.0 if self.has_green_roof else 0.0) * 0.3 +
            min(self.rooftop_units_qty / 10.0, 1.0) * 0.4 +
            min(self.fall_anchor_count / 50.0, 1.0) * 0.3
        )
        return min(score, 1.0)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'square_feet': self.square_feet,
            'stories': self.stories,
            'has_green_roof': self.has_green_roof,
            'rooftop_units_qty': self.rooftop_units_qty,
            'fall_anchor_count': self.fall_anchor_count,
            'building_type': self.building_type,
            'foundation_type': self.foundation_type,
            'structural_system': self.structural_system,
            'facade_type': self.facade_type,
            'sqft_per_story': self.sqft_per_story(),
            'complexity_score': self.complexity_score(),
        }


@dataclass
class SubJob:
    """
    Sub-job representing a temporal segment of the project.

    Implements east-first sequencing pattern with phase overlaps.
    Models the common construction pattern where:
    - Sub-jobs start sequentially (e.g., East building before West)
    - Design phases can fully overlap
    - Construction phases have ~1 month overlap

    Attributes:
        id: Unique identifier
        parent_gmp_id: Parent GMP allocation
        name: Sub-job name (e.g., "Building A - East")
        sequence_number: Ordering for east-first pattern (1 = first/east)
        building_params: Physical parameters for this sub-job scope
        design_timeline: Design phase timeline
        preconstruction_timeline: Preconstruction phase timeline
        construction_timeline: Construction phase timeline
        closeout_timeline: Closeout phase timeline
        allocated_gmp: GMP amount allocated to this sub-job
        spent_to_date: Actual cost spent to date
        current_phase: Current phase status
        percent_complete: Overall percent complete
    """

    id: UUID = field(default_factory=uuid4)
    parent_gmp_id: Optional[UUID] = None
    name: str = ""
    sequence_number: int = 0  # East-first ordering

    # Building parameters for this sub-job scope
    building_params: BuildingParameters = field(default_factory=BuildingParameters)

    # Phase timelines
    design_timeline: PhaseTimeline = field(
        default_factory=lambda: PhaseTimeline(SubJobPhase.DESIGN)
    )
    preconstruction_timeline: PhaseTimeline = field(
        default_factory=lambda: PhaseTimeline(SubJobPhase.PRECONSTRUCTION)
    )
    construction_timeline: PhaseTimeline = field(
        default_factory=lambda: PhaseTimeline(SubJobPhase.CONSTRUCTION)
    )
    closeout_timeline: PhaseTimeline = field(
        default_factory=lambda: PhaseTimeline(SubJobPhase.CLOSEOUT)
    )

    # Cost allocation
    allocated_gmp: Decimal = Decimal("0.00")
    spent_to_date: Decimal = Decimal("0.00")

    # Status
    current_phase: SubJobPhase = SubJobPhase.NOT_STARTED
    percent_complete: float = 0.0

    def overlaps_with(self, other: 'SubJob') -> Dict[str, int]:
        """
        Calculate phase overlaps with another sub-job.

        Args:
            other: Another SubJob to compare

        Returns:
            Dictionary with overlap days per phase
        """
        return {
            'design': self.design_timeline.overlap_days(other.design_timeline),
            'preconstruction': self.preconstruction_timeline.overlap_days(
                other.preconstruction_timeline
            ),
            'construction': self.construction_timeline.overlap_days(
                other.construction_timeline
            ),
            'closeout': self.closeout_timeline.overlap_days(other.closeout_timeline),
        }

    def validate_sequencing(self, predecessor: Optional['SubJob']) -> bool:
        """
        Validate sequencing constraints.

        Business rules:
        - Design phases can overlap fully
        - Construction overlap should be ~1 month (30 days +/- 10)

        Args:
            predecessor: Previous sub-job in sequence (or None if first)

        Returns:
            True if sequencing is valid
        """
        if predecessor is None:
            return True

        overlaps = self.overlaps_with(predecessor)

        # Construction overlap should be approximately 1 month
        construction_overlap = overlaps['construction']
        if not (20 <= construction_overlap <= 40):
            return False  # Outside acceptable overlap window

        return True

    def calculate_remaining_budget(self) -> Decimal:
        """Calculate remaining budget for this sub-job."""
        return self.allocated_gmp - self.spent_to_date

    def budget_burn_rate(self) -> float:
        """
        Calculate budget burn rate.

        Returns:
            Percentage of allocated GMP spent
        """
        if self.allocated_gmp == 0:
            return 0.0
        return float(self.spent_to_date / self.allocated_gmp) * 100

    def total_planned_duration(self) -> int:
        """Calculate total planned duration across all phases."""
        return (
            self.design_timeline.duration_days() +
            self.preconstruction_timeline.duration_days() +
            self.construction_timeline.duration_days() +
            self.closeout_timeline.duration_days()
        )

    def planned_end_date(self) -> Optional[date]:
        """Get the planned end date (closeout end)."""
        return self.closeout_timeline.planned_end

    def planned_start_date(self) -> Optional[date]:
        """Get the planned start date (design start)."""
        return self.design_timeline.planned_start

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'parent_gmp_id': str(self.parent_gmp_id) if self.parent_gmp_id else None,
            'name': self.name,
            'sequence_number': self.sequence_number,
            'building_params': self.building_params.to_dict(),
            'allocated_gmp': float(self.allocated_gmp),
            'spent_to_date': float(self.spent_to_date),
            'remaining_budget': float(self.calculate_remaining_budget()),
            'budget_burn_rate': self.budget_burn_rate(),
            'current_phase': self.current_phase.value,
            'percent_complete': self.percent_complete,
            'total_planned_duration': self.total_planned_duration(),
            'planned_start': (
                self.planned_start_date().isoformat()
                if self.planned_start_date() else None
            ),
            'planned_end': (
                self.planned_end_date().isoformat()
                if self.planned_end_date() else None
            ),
        }
