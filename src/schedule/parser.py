"""
Schedule Parser - Extracts structured data from schedule.csv
THIS IS THE FOUNDATION OF THE ENTIRE SYSTEM

The schedule is the PRIMARY driver of cost predictions:
- Activities define WHEN costs occur
- Phase progression determines trade activation
- S-curves derived from activity timing
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CostCurve(Enum):
    """Cost distribution patterns over time"""
    STEP = "step"              # One-time cost at start
    FRONT_LOADED = "front_loaded"  # Heavy spending early
    CONCENTRATED = "concentrated"   # Most cost in short window
    NORMAL = "normal"          # Bell curve distribution
    EXTENDED = "extended"      # Spread evenly
    BACK_LOADED = "back_loaded"    # Light early, heavy late


@dataclass
class Activity:
    """Single schedule activity with trade mapping"""
    id: str
    name: str
    start: datetime
    finish: datetime
    duration_days: int
    total_float: float

    # Derived fields
    phase: str = ""
    primary_trade: str = ""
    secondary_trades: List[str] = field(default_factory=list)
    cost_weight: float = 1.0
    is_critical: bool = False
    sub_job: str = ""  # EAST, WEST, or SHARED
    intensity_factor: float = 1.0  # Cost density multiplier

    def is_active(self, as_of: datetime) -> bool:
        """Check if activity is in progress on given date"""
        return self.start <= as_of <= self.finish

    def is_complete(self, as_of: datetime) -> bool:
        """Check if activity is complete by given date"""
        return as_of > self.finish

    def pct_complete(self, as_of: datetime) -> float:
        """Calculate % complete by given date"""
        if as_of < self.start:
            return 0.0
        if as_of > self.finish:
            return 1.0
        elapsed = (as_of - self.start).days
        return elapsed / max(1, self.duration_days)

    def days_remaining(self, as_of: datetime) -> int:
        """Days remaining from given date"""
        if as_of > self.finish:
            return 0
        return max(0, (self.finish - as_of).days)


@dataclass
class Phase:
    """Construction phase grouping activities"""
    id: str
    name: str
    start: datetime
    end: datetime
    activities: List[Activity]
    trades_active: List[str]
    cost_curve: CostCurve

    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days

    def pct_complete(self, as_of: datetime) -> float:
        if as_of < self.start:
            return 0.0
        if as_of > self.end:
            return 1.0
        return (as_of - self.start).days / max(1, self.duration_days)

    def is_active(self, as_of: datetime) -> bool:
        return self.start <= as_of <= self.end


class ScheduleParser:
    """
    Parses schedule.csv and maps activities to GMP trades.
    THIS IS THE CORE OF THE SYSTEM.

    Every dollar spent maps to a SCHEDULE ACTIVITY.
    The model learns:
    - Which activities drive which trade costs
    - How costs distribute across activity duration
    - What cost velocity to expect at each project phase
    """

    # Activity name -> Trade mapping rules
    # Pattern: (primary_trade, secondary_trades, cost_weight, intensity_factor)
    TRADE_KEYWORDS = {
        # Concrete
        r'FTG|FTGS|FOOTING': ('Concrete', [], 1.0, 1.5),
        r'SOG|SLAB': ('Concrete', [], 1.0, 1.4),
        r'POUR.*DECK|DECK.*POUR': ('Concrete', ['Structural Steel'], 0.7, 1.5),
        r'REBAR|WIRE': ('Concrete', [], 1.0, 1.2),
        r'GB|GRADE BEAM': ('Concrete', [], 1.0, 1.3),
        r'PIER': ('Concrete', [], 1.0, 1.3),

        # Structural Steel
        r'ERECT.*STEEL|STEEL|STRUCT.*STEEL': ('Structural Steel', [], 1.0, 1.4),
        r'METAL DECK|MTL DECK': ('Structural Steel', [], 1.0, 1.2),
        r'NELSON STUD|WELD.*STUD': ('Structural Steel', [], 1.0, 1.1),
        r'DUNNAGE|MECH.*SCREEN': ('Structural Steel', [], 1.0, 1.0),

        # Masonry
        r'CMU': ('Masonry', [], 1.0, 1.1),
        r'BRICK': ('Masonry', [], 1.0, 1.1),
        r'MASONRY': ('Masonry', [], 1.0, 1.0),

        # Roofing
        r'VAPOR.*BARRIER|ROOF.*VAPOR': ('Roofing', [], 1.0, 0.9),
        r'SKYLIGHT': ('Roofing', ['Aluminum & Glass'], 0.6, 1.3),
        r'PVC|INSULATION.*ROOF|MAIN ROOF': ('Roofing', [], 1.0, 1.2),
        r'GREEN ROOF': ('Roofing', ['Landscaping'], 0.85, 1.4),
        r'PARAPET|COPING': ('Roofing', ['Masonry'], 0.7, 1.0),
        r'ROOF DRAIN': ('Plumbing & H.V.A.C', [], 1.0, 0.8),

        # Windows/Glazing
        r'WINDOW|SF WINDOW|STOREFRONT': ('Aluminum & Glass', [], 1.0, 1.3),
        r'CURTAIN WALL': ('Aluminum & Glass', [], 1.0, 1.4),

        # Waterproofing
        r'AVB|AIR.*VAPOR': ('Waterproofing', [], 1.0, 0.9),
        r'WATERPROOF': ('Waterproofing', [], 1.0, 1.0),
        r'FOUNDATION.*WATERPROOF': ('Waterproofing', [], 1.0, 1.1),

        # Framing/Carpentry
        r'FRAME.*WALL|INT.*WALL|WALL.*FRAME': ('Rough Carpentry, Drywall, Ceilings', [], 1.0, 0.9),
        r'SHEATH': ('Rough Carpentry, Drywall, Ceilings', [], 1.0, 0.8),
        r'CEIL|CORRIDOR CEIL|CEILING': ('Rough Carpentry, Drywall, Ceilings', [], 1.0, 0.9),
        r'INSULATION|THERM.*INSUL|WALL INSUL': ('Rough Carpentry, Drywall, Ceilings', [], 0.8, 0.8),
        r'BLOCKING|CLOSE-IN|PUNCHLIST': ('Rough Carpentry, Drywall, Ceilings', [], 1.0, 0.7),
        r'DRYWALL': ('Rough Carpentry, Drywall, Ceilings', [], 1.0, 0.9),

        # MEP - Plumbing/HVAC
        r'PLUMB|U\.G\. PLUMB|UNDERGROUND PLUMB': ('Plumbing & H.V.A.C', [], 1.0, 1.0),
        r'DUCT|DUCTWORK': ('Plumbing & H.V.A.C', [], 1.0, 1.0),
        r'HVAC|PIPING|HVAC PIPING': ('Plumbing & H.V.A.C', [], 1.0, 1.1),
        r'ACCU|DOAS|AHU': ('Plumbing & H.V.A.C', [], 1.0, 1.3),
        r'ROOFTOP.*UNIT|RTU': ('Plumbing & H.V.A.C', [], 1.0, 1.4),

        # MEP - Electrical
        r'ELEC|ELECTRICAL|COMM RI': ('Electrical & Fire Alarm', [], 1.0, 1.0),
        r'PANEL|SWITCHGEAR': ('Electrical & Fire Alarm', [], 1.0, 1.2),
        r'TRANSFORMER': ('Electrical & Fire Alarm', [], 1.0, 1.3),

        # MEP - Low Voltage
        r'AV|SECURITY|PULL.*CABLE': ('Low Voltage', [], 1.0, 0.9),
        r'DATA|COMM|TELECOM': ('Low Voltage', [], 1.0, 0.9),

        # MEP - Fire Protection
        r'FS MAIN|FS LAT|FIRE.*MAIN|SPRINKLER': ('Fire Protection', [], 1.0, 1.0),
        r'STANDPIPE': ('Fire Protection', [], 1.0, 1.1),

        # Sitework
        r'MOBILIZE': ('General Requirements', ['Sitework'], 0.7, 1.0),
        r'SURVEY|LOD': ('General Requirements', [], 1.0, 0.5),
        r'EXC|EXCAVAT': ('Sitework', ['Concrete'], 0.7, 1.1),
        r'DRILL.*PILE|GROUT|MICROPILE': ('Sitework', [], 1.0, 1.3),
        r'LAG|SOE': ('Sitework', [], 1.0, 1.2),
        r'TOPSOIL|GRADE|GRADING|CUT|FILL': ('Sitework', [], 1.0, 1.0),
        r'COMPACT|FABRIC|STONE': ('Sitework', [], 1.0, 0.9),
        r'BACKFILL': ('Sitework', [], 1.0, 0.8),
        r'DRIVEWAY|TEMP DRIVE': ('Sitework', [], 1.0, 0.9),
        r'RETAINING|BIOPLANTER': ('Sitework', ['Masonry'], 0.5, 1.0),
        r'SIDEWALK|STAIR': ('Sitework', ['Concrete'], 0.6, 1.0),
        r'PLAYGROUND|PLAY STRUCTURE|PIP': ('Sitework', [], 1.0, 1.2),
        r'HAND RAIL|RAILING': ('Sitework', [], 0.6, 0.8),
        r'SHADE STRUCTURE': ('Sitework', [], 1.0, 1.1),
        r'FENCE|FENCING': ('Sitework', [], 1.0, 0.8),

        # Landscaping
        r'ROOT PRUNE|TREE.*PROTECT': ('Landscaping', ['Sitework'], 0.6, 0.6),
        r'SOD|PLANTING|PLANT': ('Landscaping', [], 1.0, 1.0),
        r'TOPSOIL.*PLACE|PLACE.*TOPSOIL': ('Landscaping', [], 1.0, 0.9),
        r'MULCH': ('Landscaping', [], 1.0, 0.7),
        r'IRRIGATION': ('Landscaping', [], 1.0, 1.0),

        # Inspections (minimal cost)
        r'INSPECT': ('General Requirements', [], 1.0, 0.2),

        # Cure time (almost no cost)
        r'CURE|CURE TIME': ('General Requirements', [], 1.0, 0.05),

        # Demolition
        r'DEMO|DEMOLITION': ('Site Demolition', [], 1.0, 1.0),
    }

    # Sub-job indicators
    EAST_INDICATORS = [r'NORTH', r'N\.', r'N 1/2', r'1ST FL', r'1st FL', r'EAST', r'E\.']
    WEST_INDICATORS = [r'SOUTH', r'S\.', r'WS\.', r'WC\.', r'2ND FL', r'2nd FL', r'CAFETERIA', r'WEST', r'W\.']

    # Phase order and cost curves
    PHASE_ORDER = ['MOBILIZATION', 'SITE_PREP', 'FOUNDATION', 'STRUCTURE',
                   'ENVELOPE', 'MEP_ROUGH', 'FINISHES', 'SITE_FINISHES', 'OTHER']

    PHASE_COST_CURVES = {
        'MOBILIZATION': CostCurve.STEP,
        'SITE_PREP': CostCurve.FRONT_LOADED,
        'FOUNDATION': CostCurve.CONCENTRATED,
        'STRUCTURE': CostCurve.NORMAL,
        'ENVELOPE': CostCurve.EXTENDED,
        'MEP_ROUGH': CostCurve.EXTENDED,
        'FINISHES': CostCurve.BACK_LOADED,
        'SITE_FINISHES': CostCurve.BACK_LOADED,
        'OTHER': CostCurve.EXTENDED,
    }

    def __init__(self, schedule_df: pd.DataFrame):
        """
        Initialize parser with schedule DataFrame.

        Args:
            schedule_df: DataFrame with columns like 'Activity ID', 'Activity Name',
                        'Start', 'Finish', 'Duration', 'Total Float'
        """
        self.raw_schedule = schedule_df
        self.activities: List[Activity] = []
        self.phases: List[Phase] = []
        self.project_start: Optional[datetime] = None
        self.project_end: Optional[datetime] = None

        self._parse()

    def _parse_date(self, date_str) -> Optional[datetime]:
        """Parse date string, handling 'A' suffix for actuals"""
        if pd.isna(date_str):
            return None
        date_str = str(date_str).strip().rstrip(' A')

        # Try multiple date formats
        formats = ['%d-%b-%y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _map_to_trade(self, activity_name: str) -> Tuple[str, List[str], float, float]:
        """
        Map activity name to GMP trade(s).

        Returns:
            (primary_trade, secondary_trades, cost_weight, intensity_factor)
        """
        name_upper = activity_name.upper()

        for pattern, (primary, secondary, weight, intensity) in self.TRADE_KEYWORDS.items():
            if re.search(pattern, name_upper):
                return primary, secondary, weight, intensity

        # Default to General Requirements
        return 'General Requirements', [], 0.5, 0.5

    def _determine_sub_job(self, activity_name: str) -> str:
        """Determine if activity is EAST, WEST, or SHARED"""
        name_upper = activity_name.upper()

        for pattern in self.EAST_INDICATORS:
            if re.search(pattern, name_upper):
                return 'EAST'

        for pattern in self.WEST_INDICATORS:
            if re.search(pattern, name_upper):
                return 'WEST'

        return 'SHARED'  # Applies to both

    def _determine_phase(self, activity: Activity) -> str:
        """Determine construction phase from activity"""
        name_upper = activity.name.upper()
        id_upper = activity.id.upper()

        # Mobilization
        if any(x in name_upper for x in ['MOBILIZE', 'SURVEY', 'LOD']):
            return 'MOBILIZATION'

        # Site Prep
        if id_upper.startswith('SITE-0') or id_upper.startswith('SITE0'):
            return 'SITE_PREP'
        if any(x in name_upper for x in ['TREE', 'ROOT', 'SOE', 'LAG', 'DRILL', 'PILE']):
            return 'SITE_PREP'

        # Foundation
        if any(x in name_upper for x in ['FTG', 'FTGS', 'SOG', 'PIER', 'BACKFILL', 'GRADE BEAM', 'GB']):
            return 'FOUNDATION'

        # Structure
        if any(x in name_upper for x in ['STEEL', 'CMU', 'DECK', 'POUR', 'FRAME', 'ERECT']):
            return 'STRUCTURE'

        # Envelope
        if any(x in name_upper for x in ['ROOF', 'BRICK', 'WINDOW', 'AVB', 'PARAPET', 'CURTAIN']):
            return 'ENVELOPE'

        # MEP Rough-in
        if any(x in name_upper for x in ['PLUMB', 'DUCT', 'ELEC', 'FS', 'HVAC', 'PIPING', 'FIRE']):
            return 'MEP_ROUGH'

        # Finishes
        if any(x in name_upper for x in ['CEIL', 'INSUL', 'INSPECT', 'CLOSE', 'DRYWALL', 'PAINT']):
            return 'FINISHES'

        # Site Finishes
        if id_upper.startswith('SITE-2') or id_upper.startswith('SITE2'):
            return 'SITE_FINISHES'
        if any(x in name_upper for x in ['PLAYGROUND', 'SOD', 'PLANT', 'SIDEWALK', 'LANDSCAPE']):
            return 'SITE_FINISHES'

        return 'OTHER'

    def _parse(self):
        """Parse schedule into structured activities"""

        # Try to find column names (handle different naming conventions)
        col_mappings = {
            'activity_id': ['Activity ID', 'activity_id', 'ActivityID', 'ID'],
            'activity_name': ['Activity Name', 'activity_name', 'ActivityName', 'Name', 'Description'],
            'start': ['Start', 'start_date', 'StartDate', 'start'],
            'finish': ['Finish', 'finish_date', 'FinishDate', 'finish', 'End'],
            'duration': ['Duration', 'duration_days', 'DurationDays', 'duration'],
            'float': ['Total Float', 'total_float', 'TotalFloat', 'Float', 'float']
        }

        def find_column(df, options):
            for opt in options:
                if opt in df.columns:
                    return opt
            return None

        cols = {k: find_column(self.raw_schedule, v) for k, v in col_mappings.items()}

        for _, row in self.raw_schedule.iterrows():
            start = self._parse_date(row.get(cols['start']) if cols['start'] else None)
            finish = self._parse_date(row.get(cols['finish']) if cols['finish'] else None)

            if not start or not finish:
                continue

            activity_id = str(row.get(cols['activity_id'], '') if cols['activity_id'] else '')
            activity_name = str(row.get(cols['activity_name'], '') if cols['activity_name'] else '')

            duration_val = row.get(cols['duration'], 1) if cols['duration'] else 1
            duration = int(duration_val) if pd.notna(duration_val) else max(1, (finish - start).days)

            float_val = row.get(cols['float'], 0) if cols['float'] else 0
            total_float = float(float_val) if pd.notna(float_val) else 0.0

            primary_trade, secondary_trades, cost_weight, intensity = self._map_to_trade(activity_name)
            sub_job = self._determine_sub_job(activity_name)

            activity = Activity(
                id=activity_id,
                name=activity_name,
                start=start,
                finish=finish,
                duration_days=duration,
                total_float=total_float,
                primary_trade=primary_trade,
                secondary_trades=secondary_trades,
                cost_weight=cost_weight,
                is_critical=(total_float == 0),
                sub_job=sub_job,
                intensity_factor=intensity
            )

            activity.phase = self._determine_phase(activity)
            self.activities.append(activity)

        # Set project timeline
        if self.activities:
            self.project_start = min(a.start for a in self.activities)
            self.project_end = max(a.finish for a in self.activities)

        # Build phases
        self._build_phases()

        logger.info(f"Parsed {len(self.activities)} activities into {len(self.phases)} phases")
        logger.info(f"Project timeline: {self.project_start} to {self.project_end}")

    def _build_phases(self):
        """Group activities into phases"""
        phase_activities: Dict[str, List[Activity]] = {}

        for activity in self.activities:
            if activity.phase not in phase_activities:
                phase_activities[activity.phase] = []
            phase_activities[activity.phase].append(activity)

        for phase_id in self.PHASE_ORDER:
            if phase_id not in phase_activities:
                continue

            acts = phase_activities[phase_id]
            trades = list(set(a.primary_trade for a in acts))

            phase = Phase(
                id=phase_id,
                name=phase_id.replace('_', ' ').title(),
                start=min(a.start for a in acts),
                end=max(a.finish for a in acts),
                activities=acts,
                trades_active=trades,
                cost_curve=self.PHASE_COST_CURVES.get(phase_id, CostCurve.EXTENDED)
            )
            self.phases.append(phase)

    def get_active_activities(self, as_of: datetime) -> List[Activity]:
        """Get all activities in progress on a date"""
        return [a for a in self.activities if a.is_active(as_of)]

    def get_activities_for_trade(self, trade_name: str) -> List[Activity]:
        """Get all activities mapped to a trade"""
        return [a for a in self.activities
                if a.primary_trade == trade_name or trade_name in a.secondary_trades]

    def get_current_phase(self, as_of: datetime) -> Optional[Phase]:
        """Get the primary active phase"""
        for phase in self.phases:
            if phase.is_active(as_of):
                return phase
        return None

    def get_active_phases(self, as_of: datetime) -> List[Phase]:
        """Get all active phases (can overlap)"""
        return [p for p in self.phases if p.is_active(as_of)]

    def project_pct_complete(self, as_of: datetime) -> float:
        """Project % complete by date"""
        if not self.project_start or not self.project_end:
            return 0.0
        total = (self.project_end - self.project_start).days
        elapsed = (as_of - self.project_start).days
        return max(0, min(1, elapsed / max(1, total)))

    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of activities by trade"""
        trade_data = {}

        for activity in self.activities:
            trade = activity.primary_trade
            if trade not in trade_data:
                trade_data[trade] = {
                    'trade': trade,
                    'activity_count': 0,
                    'total_duration_days': 0,
                    'earliest_start': None,
                    'latest_finish': None,
                    'critical_activities': 0
                }

            trade_data[trade]['activity_count'] += 1
            trade_data[trade]['total_duration_days'] += activity.duration_days
            trade_data[trade]['critical_activities'] += 1 if activity.is_critical else 0

            if trade_data[trade]['earliest_start'] is None or activity.start < trade_data[trade]['earliest_start']:
                trade_data[trade]['earliest_start'] = activity.start
            if trade_data[trade]['latest_finish'] is None or activity.finish > trade_data[trade]['latest_finish']:
                trade_data[trade]['latest_finish'] = activity.finish

        return pd.DataFrame(list(trade_data.values()))
