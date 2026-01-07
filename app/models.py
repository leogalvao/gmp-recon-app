"""
Database models and SQLAlchemy setup for GMP Reconciliation App.
All monetary values stored as integer cents to avoid float drift.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import enum

DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Region(enum.Enum):
    WEST = "West"
    EAST = "East"
    BOTH = "Both"


class ForecastBasis(enum.Enum):
    ACTUALS_ONLY = "actuals_only"
    ACTUALS_PLUS_COMMITMENTS = "actuals_plus_commitments"


class EACMode(enum.Enum):
    MAX = "max"
    MODEL = "model"
    COMMITMENTS = "commitments"


class BudgetToGMP(Base):
    """Mapping from Budget rows to GMP divisions."""
    __tablename__ = "budget_to_gmp"

    id = Column(Integer, primary_key=True, index=True)
    budget_code = Column(String(50), index=True)
    cost_code_tier2 = Column(String(100))
    gmp_division = Column(String(200), index=True)
    side = Column(String(4), default="BOTH", index=True)  # EAST, WEST, BOTH
    confidence = Column(Float, default=1.0)  # 0.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DirectToBudget(Base):
    """Mapping from Direct Cost rows to Budget codes."""
    __tablename__ = "direct_to_budget"

    id = Column(Integer, primary_key=True, index=True)
    cost_code = Column(String(50), index=True)
    name = Column(String(200))
    budget_code = Column(String(50), index=True)
    side = Column(String(4), default="BOTH", index=True)  # EAST, WEST, BOTH
    confidence = Column(Float, default=1.0)
    method = Column(String(30), default='manual')  # manual, user_confirmed, base_code_exact, fuzzy_match, bulk_accept
    vendor_normalized = Column(String(255), nullable=True, index=True)  # For pattern matching
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Allocation(Base):
    """Regional allocation splits for cost codes."""
    __tablename__ = "allocations"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, index=True)  # Cost Code or Budget Code
    region = Column(String(10), default="Both")  # West, East, Both
    pct_west = Column(Float, default=0.5)
    pct_east = Column(Float, default=0.5)
    confirmed = Column(Boolean, default=False)  # User confirmed the split
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Settings(Base):
    """Global application settings."""
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    as_of_date = Column(DateTime, nullable=True)  # None = auto (max transaction date)
    forecast_basis = Column(String(50), default="actuals_plus_commitments")
    eac_mode_when_commitments = Column(String(20), default="max")
    gmp_scope_notes = Column(Text, nullable=True)
    gmp_scope_confirmed = Column(Boolean, default=False)
    # Breakdown and schedule integration settings
    use_breakdown_allocations = Column(Boolean, default=True)  # Use breakdown.csv for E/W splits
    use_schedule_forecast = Column(Boolean, default=False)  # Use schedule progress for EAC
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Run(Base):
    """Audit trail for reconciliation runs."""
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_type = Column(String(50))  # recompute, nightly_train, file_change
    status = Column(String(20))  # pending, running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    as_of_date = Column(DateTime, nullable=True)
    forecast_basis = Column(String(50), nullable=True)
    file_hashes = Column(Text, nullable=True)  # JSON of file hashes
    notes = Column(Text, nullable=True)


class Duplicate(Base):
    """Suspected duplicate direct cost entries."""
    __tablename__ = "duplicates"
    
    id = Column(Integer, primary_key=True, index=True)
    direct_cost_row_id = Column(Integer, index=True)  # Row index in direct costs
    group_id = Column(Integer, index=True)  # Group ID for related duplicates
    method = Column(String(50))  # exact, fuzzy_vendor, fuzzy_amount
    score = Column(Float)  # Confidence score 0-1
    resolved = Column(Boolean, default=False)
    excluded_from_actuals = Column(Boolean, default=False)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)


class MappingAudit(Base):
    """Audit trail for mapping changes."""
    __tablename__ = "mapping_audit"

    id = Column(Integer, primary_key=True, index=True)
    table_name = Column(String(50))  # budget_to_gmp, direct_to_budget, allocations
    record_id = Column(Integer)
    action = Column(String(20))  # create, update, delete
    old_value = Column(Text, nullable=True)  # JSON
    new_value = Column(Text, nullable=True)  # JSON
    user = Column(String(100), default="system")
    timestamp = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Enhanced Direct Cost → Budget Mapping Tables (Phase 2)
# =============================================================================

class MappingFeedback(Base):
    """
    Stores learned mapping patterns from user interactions.
    Used for feedback loop: when user accepts/overrides suggestions,
    patterns are recorded to improve future suggestions.
    """
    __tablename__ = "mapping_feedback"

    id = Column(Integer, primary_key=True, index=True)
    vendor_normalized = Column(String(255), nullable=False, index=True)
    name_prefix = Column(String(50), nullable=False, index=True)  # First 20 chars, lowercased
    budget_code = Column(String(50), nullable=False, index=True)
    was_override = Column(Boolean, default=False)  # True if user picked different than suggested
    suggested_budget_code = Column(String(50), nullable=True)  # What system suggested (if override)
    confidence_at_suggestion = Column(Float, nullable=True)  # Score when suggestion was made
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BudgetMatchStats(Base):
    """
    Aggregated statistics for budget codes used in matching.
    Provides fast lookup of historical match counts and trust scores.
    """
    __tablename__ = "budget_match_stats"

    budget_code = Column(String(50), primary_key=True)
    total_matches = Column(Integer, default=0)  # Times this budget was selected
    override_count = Column(Integer, default=0)  # Times user picked something else when this was suggested
    trust_score = Column(Float, default=1.0)  # Decays with overrides, used in tie-breaking
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SuggestionCache(Base):
    """
    Precomputed match suggestions for direct cost rows.
    Enables fast page loads by caching expensive fuzzy matching results.
    Marked stale when related data changes, triggering background recompute.
    """
    __tablename__ = "suggestion_cache"

    direct_cost_id = Column(Integer, primary_key=True)  # Row ID from direct costs DataFrame
    suggestions = Column(Text, nullable=False)  # JSON: [{budget_code, score, description}, ...]
    top_score = Column(Float, default=0.0, index=True)  # Denormalized for filtering by confidence
    computed_at = Column(DateTime, default=datetime.utcnow)
    stale = Column(Boolean, default=False, index=True)  # True when cache needs refresh


class GMPAllocationOverride(Base):
    """
    Manual overrides for GMP division allocation amounts.
    Allows users to directly edit Assigned East/West values.
    Stores amounts in cents for precision.
    """
    __tablename__ = "gmp_allocation_overrides"

    id = Column(Integer, primary_key=True, index=True)
    gmp_division = Column(String(200), unique=True, index=True)
    amount_west_cents = Column(Integer, nullable=True)  # Manual override for West, null = use computed
    amount_east_cents = Column(Integer, nullable=True)  # Manual override for East, null = use computed
    notes = Column(Text, nullable=True)
    created_by = Column(String(100), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AllocationChangeLog(Base):
    """
    Audit log for allocation changes.
    Tracks all modifications to assigned amounts for traceability.
    """
    __tablename__ = "allocation_change_log"

    id = Column(Integer, primary_key=True, index=True)
    gmp_division = Column(String(200), index=True)
    field_changed = Column(String(50))  # amount_west, amount_east
    old_value_cents = Column(Integer, nullable=True)
    new_value_cents = Column(Integer, nullable=True)
    change_reason = Column(String(200), nullable=True)
    changed_by = Column(String(100), default="user")
    changed_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Forecasting Module Tables (Phase 3)
# =============================================================================

class ForecastMethod(enum.Enum):
    """Supported forecasting methods."""
    EVM = "evm"                  # Earned Value Management
    PERT = "pert"                # Three-point estimating
    PARAMETRIC = "parametric"    # Quantity x Unit Rate x Factor
    ML_LINEAR = "ml_linear"      # Existing LinearRegression model
    MANUAL = "manual"            # User override


class ForecastConfig(Base):
    """
    Stores forecasting method selection and parameters per GMP line.
    Each GMP division can have its own forecasting approach.
    """
    __tablename__ = "forecast_config"

    id = Column(Integer, primary_key=True, index=True)
    gmp_division = Column(String(200), unique=True, index=True, nullable=False)
    method = Column(String(30), default='evm')  # ForecastMethod.value

    # EVM parameters
    evm_performance_factor = Column(Float, default=1.0)  # CPI adjustment factor

    # PERT parameters (stored as cents)
    pert_optimistic_cents = Column(Integer, nullable=True)
    pert_most_likely_cents = Column(Integer, nullable=True)
    pert_pessimistic_cents = Column(Integer, nullable=True)

    # Parametric parameters
    param_quantity = Column(Float, nullable=True)
    param_unit_rate_cents = Column(Integer, nullable=True)
    param_complexity_factor = Column(Float, default=1.0)

    # Distribution method for remaining cost
    distribution_method = Column(String(20), default='linear')  # linear, front_loaded, back_loaded

    # Completion date override (null = use project default)
    completion_date = Column(DateTime, nullable=True)

    # Metadata
    is_locked = Column(Boolean, default=False)  # Prevent auto-updates
    notes = Column(Text, nullable=True)
    created_by = Column(String(100), default="system")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ForecastSnapshot(Base):
    """
    Point-in-time forecast snapshot for a GMP division.
    Stores computed EAC, method used, and confidence metrics.
    New snapshot created on each recalculation, previous marked superseded.
    """
    __tablename__ = "forecast_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    gmp_division = Column(String(200), index=True, nullable=False)
    snapshot_date = Column(DateTime, default=datetime.utcnow, index=True)

    # Forecast values (in cents)
    bac_cents = Column(Integer, nullable=False)       # Budget at Completion (GMP amount)
    ac_cents = Column(Integer, nullable=False)        # Actual Cost to date
    ev_cents = Column(Integer, nullable=True)         # Earned Value (for EVM, if tracked)
    eac_cents = Column(Integer, nullable=False)       # Estimate at Completion
    eac_west_cents = Column(Integer, nullable=False)  # EAC for West region
    eac_east_cents = Column(Integer, nullable=False)  # EAC for East region
    etc_cents = Column(Integer, nullable=False)       # Estimate to Complete (EAC - AC)
    var_cents = Column(Integer, nullable=False)       # Variance (BAC - EAC), negative = over budget

    # Performance indices (for EVM)
    cpi = Column(Float, nullable=True)   # Cost Performance Index (EV/AC)
    spi = Column(Float, nullable=True)   # Schedule Performance Index (if available)

    # Method and confidence
    method = Column(String(30), nullable=False)
    confidence_score = Column(Float, default=0.5)     # 0.0-1.0
    confidence_band = Column(String(20), default='medium')  # low, medium, high

    # Method explanation (plain language)
    explanation = Column(Text, nullable=True)

    # Snapshot lifecycle
    is_current = Column(Boolean, default=True, index=True)
    superseded_by_id = Column(Integer, nullable=True)
    trigger = Column(String(50), default='manual')    # manual, transaction, mapping, schedule
    created_at = Column(DateTime, default=datetime.utcnow)


class ForecastPeriod(Base):
    """
    Time-bucketed forecast data for a GMP division.
    Supports weekly (ISO Mon-Sun) and monthly (calendar) granularity.
    """
    __tablename__ = "forecast_periods"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(Integer, index=True, nullable=False)  # FK to ForecastSnapshot
    gmp_division = Column(String(200), index=True, nullable=False)

    # Period identification
    granularity = Column(String(10), nullable=False)  # 'weekly' or 'monthly'
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    period_label = Column(String(20), nullable=False)  # e.g., "2026-W01" or "2026-01"
    period_number = Column(Integer, nullable=False)    # Sequential: 1, 2, 3...
    iso_week = Column(Integer, nullable=True)          # ISO week number (1-53)
    iso_year = Column(Integer, nullable=True)          # ISO year

    # Period type relative to as_of_date
    period_type = Column(String(20), nullable=False)  # 'past', 'current', 'future'

    # Amounts (in cents)
    actual_cents = Column(Integer, default=0)         # Actual spend in period (past/current)
    forecast_cents = Column(Integer, default=0)       # Forecasted spend (current/future)
    blended_cents = Column(Integer, default=0)        # Actual + forecast for current period
    cumulative_cents = Column(Integer, default=0)     # Running total through this period

    # Regional split
    actual_west_cents = Column(Integer, default=0)
    actual_east_cents = Column(Integer, default=0)
    forecast_west_cents = Column(Integer, default=0)
    forecast_east_cents = Column(Integer, default=0)

    # For week spanning months (proportional allocation)
    span_allocation_factor = Column(Float, default=1.0)  # 1.0 = full period, <1.0 = partial

    created_at = Column(DateTime, default=datetime.utcnow)


class ForecastAuditLog(Base):
    """
    Audit trail for forecast configuration and snapshot changes.
    Follows pattern of AllocationChangeLog.
    """
    __tablename__ = "forecast_audit_log"

    id = Column(Integer, primary_key=True, index=True)
    gmp_division = Column(String(200), index=True, nullable=False)
    action = Column(String(30), nullable=False)       # method_change, param_update, refresh, lock, unlock
    field_changed = Column(String(50), nullable=True)
    old_value = Column(Text, nullable=True)           # JSON
    new_value = Column(Text, nullable=True)           # JSON
    previous_eac_cents = Column(Integer, nullable=True)
    new_eac_cents = Column(Integer, nullable=True)
    change_reason = Column(String(200), nullable=True)
    changed_by = Column(String(100), default="system")
    changed_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Side Configuration (East/West/Both Phase Assignment)
# =============================================================================

class SideConfiguration(Base):
    """
    Configuration for project sides (East/West/Both).
    Stores timeline boundaries and allocation weights for each side.

    Timeline:
    - East: ends July 31, 2025
    - West: starts June 1, 2025 (1-month overlap), ends July 31, 2026
    - Both: always active, represents shared costs
    """
    __tablename__ = "side_configuration"

    id = Column(Integer, primary_key=True, index=True)
    side = Column(String(4), unique=True, nullable=False, index=True)  # EAST, WEST, BOTH
    display_name = Column(String(20), nullable=False)  # East, West, Both
    start_date = Column(DateTime, nullable=True)       # NULL = no start constraint
    end_date = Column(DateTime, nullable=True)         # NULL = no end constraint
    is_active = Column(Boolean, default=True, index=True)
    allocation_weight = Column(Float, default=0.5)     # Weight for "Both" allocation split
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =============================================================================
# GMP Budget Breakdown (East/West Funding Splits)
# =============================================================================

class GMPBudgetBreakdown(Base):
    """
    Owner's East/West funding breakdown per cost code.
    Imported from breakdown.csv and fuzzy-matched to GMP divisions.
    Used for penny-perfect allocation of actuals and forecasts.
    """
    __tablename__ = 'gmp_budget_breakdown'

    id = Column(Integer, primary_key=True, index=True)
    cost_code_description = Column(String(200), nullable=False)
    gmp_division = Column(String(200), index=True)  # Matched GMP division (null if unmatched)
    gmp_sov_cents = Column(Integer, nullable=False)  # Total SOV in cents
    east_funded_cents = Column(Integer, default=0)
    west_funded_cents = Column(Integer, default=0)
    pct_east = Column(Float, nullable=False)  # Derived: east/sov (0.0 - 1.0)
    pct_west = Column(Float, nullable=False)  # Derived: west/sov (0.0 - 1.0)
    match_score = Column(Integer, default=0)  # Fuzzy match confidence 0-100
    source_file = Column(String(100))
    imported_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Schedule Activities and Mapping (Gantt/P6 Integration)
# =============================================================================

class ScheduleActivity(Base):
    """
    Project schedule activities from Gantt/P6 export.
    Tracks task progress for schedule-based EAC calculation.
    """
    __tablename__ = 'schedule_activities'

    id = Column(Integer, primary_key=True, index=True)
    row_number = Column(Integer)
    task_name = Column(String(500), nullable=False)
    source_uid = Column(String(100), unique=True, nullable=True)  # P6 GUID
    activity_id = Column(String(50), index=True)
    wbs = Column(String(100), index=True)
    pct_complete = Column(Integer, default=0)  # 0-100
    imported_at = Column(DateTime, default=datetime.utcnow)
    source_file = Column(String(100))

    # Relationship to mappings
    mappings = relationship("ScheduleToGMPMapping", back_populates="activity", cascade="all, delete-orphan")


class ScheduleToGMPMapping(Base):
    """
    Many-to-many: Schedule activities → GMP divisions with weights.
    Allows a single activity to contribute to multiple GMP divisions.
    Weights should sum to 1.0 for each activity.
    """
    __tablename__ = 'schedule_to_gmp_mapping'

    id = Column(Integer, primary_key=True, index=True)
    schedule_activity_id = Column(Integer, ForeignKey('schedule_activities.id'), nullable=False)
    gmp_division = Column(String(200), nullable=False, index=True)
    weight = Column(Float, default=1.0)  # 0.0-1.0, should sum to 1.0 per activity
    created_by = Column(String(100), default='system')
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to activity
    activity = relationship("ScheduleActivity", back_populates="mappings")

    __table_args__ = (
        UniqueConstraint('schedule_activity_id', 'gmp_division', name='uq_schedule_gmp'),
    )


def init_db():
    """Initialize the database and create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Database session dependency for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize default settings if not exists
def ensure_default_settings():
    db = SessionLocal()
    try:
        settings = db.query(Settings).first()
        if not settings:
            settings = Settings(
                forecast_basis="actuals_plus_commitments",
                eac_mode_when_commitments="max"
            )
            db.add(settings)
            db.commit()
    finally:
        db.close()
