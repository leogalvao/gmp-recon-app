"""
Database models and SQLAlchemy setup for GMP Reconciliation App.
All monetary values stored as integer cents to avoid float drift.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Date, Text, ForeignKey, UniqueConstraint
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


class Zone(enum.Enum):
    """Spatial zone for cost allocation (alias for Region for spec compliance)."""
    EAST = "East"
    WEST = "West"
    SHARED = "Shared"  # Maps to BOTH


class ChangeOrderStatus(enum.Enum):
    """Status of a change order."""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"


class AllocationMethod(enum.Enum):
    """How to handle costs mapped to SHARED budgets."""
    DIRECT = "direct"         # Allocate to single zone
    SPLIT_50_50 = "split_50_50"  # Split evenly between zones


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

    # Start and completion date overrides (null = use data-driven defaults)
    start_date = Column(DateTime, nullable=True)
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
# Project Entity (Level 0 in Hierarchy)
# =============================================================================

class Project(Base):
    """
    Top-level project entity.
    All GMP allocations, budgets, and costs belong to a project.
    """
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)  # External UUID
    name = Column(String(200), nullable=False)
    code = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    version_id = Column(Integer, default=1)  # Optimistic locking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    gmps = relationship("GMP", back_populates="project", cascade="all, delete-orphan")
    training_rounds = relationship("TrainingRound", back_populates="project", cascade="all, delete-orphan")


# =============================================================================
# GMP Entity (Level 1 - The Funding Ceiling)
# =============================================================================

class GMP(Base):
    """
    Guaranteed Maximum Price allocation.
    The funding source, split by Division AND Zone.
    INVARIANT: Σ(Budget.current_amount) <= (GMP.original + Σ(ApprovedCOs.amount))
    """
    __tablename__ = "gmp_entities"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)  # External UUID
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False, index=True)
    division = Column(String(200), nullable=False, index=True)  # CSI Division (e.g., '03-Concrete')
    zone = Column(String(10), nullable=False, index=True)  # EAST, WEST, SHARED
    original_amount_cents = Column(Integer, nullable=False)  # Immutable base contract value
    description = Column(Text, nullable=True)
    version_id = Column(Integer, default=1)  # Optimistic locking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="gmps")
    budgets = relationship("BudgetEntity", back_populates="gmp", cascade="all, delete-orphan")
    change_orders = relationship("ChangeOrder", back_populates="gmp", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('project_id', 'division', 'zone', name='uq_project_division_zone'),
    )

    @property
    def approved_change_order_total_cents(self) -> int:
        """Sum of approved change orders."""
        return sum(co.amount_cents for co in self.change_orders
                   if co.status == ChangeOrderStatus.APPROVED.value)

    @property
    def authorized_amount_cents(self) -> int:
        """Original + Approved Change Orders."""
        return self.original_amount_cents + self.approved_change_order_total_cents


# =============================================================================
# Budget Entity (Level 2 - The Plan)
# =============================================================================

class BudgetEntity(Base):
    """
    Allocated bucket of money. Must belong to a Zone.
    INVARIANT: Budget.zone MUST MATCH GMP.zone (SPATIAL_INTEGRITY)
    """
    __tablename__ = "budget_entities"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)  # External UUID
    gmp_id = Column(Integer, ForeignKey('gmp_entities.id'), nullable=False, index=True)
    cost_code = Column(String(50), nullable=False, index=True)  # Must match Schedule Activity Code
    description = Column(String(500), nullable=True)
    zone = Column(String(10), nullable=True, index=True)  # Nullable on ingestion, assigned via UI
    current_budget_cents = Column(Integer, nullable=False, default=0)
    committed_cents = Column(Integer, default=0)  # Committed amount (contracts)
    version_id = Column(Integer, default=1)  # Optimistic locking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    gmp = relationship("GMP", back_populates="budgets")
    direct_costs = relationship("DirectCostEntity", back_populates="budget")


# =============================================================================
# Change Order Entity (Level 3b - The ONLY way to adjust GMP ceiling)
# =============================================================================

class ChangeOrder(Base):
    """
    Contract modification. The ONLY way to increase/decrease the GMP ceiling.
    Positive amount adds to scope, negative deducts.
    """
    __tablename__ = "change_orders"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)  # External UUID
    gmp_id = Column(Integer, ForeignKey('gmp_entities.id'), nullable=False, index=True)
    number = Column(String(50), nullable=False, index=True)  # CO Number (e.g., "CO-001")
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default='draft')  # draft, pending, approved
    amount_cents = Column(Integer, nullable=False)  # Can be positive or negative
    requested_date = Column(Date, nullable=True)
    approved_date = Column(Date, nullable=True)
    approved_by = Column(String(100), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    version_id = Column(Integer, default=1)  # Optimistic locking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    gmp = relationship("GMP", back_populates="change_orders")

    __table_args__ = (
        UniqueConstraint('gmp_id', 'number', name='uq_gmp_co_number'),
    )


# =============================================================================
# Direct Cost Entity (Level 3 - The Actuals)
# =============================================================================

class DirectCostEntity(Base):
    """
    Actual transaction/cost incurred on the project.
    INVARIANT: Payable = GrossAmount - RetainageHeld
    """
    __tablename__ = "direct_cost_entities"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)  # External UUID
    mapped_budget_id = Column(Integer, ForeignKey('budget_entities.id'), nullable=True, index=True)
    source_row_id = Column(Integer, nullable=True)  # Row from source file
    vendor_name = Column(String(255), nullable=True)
    vendor_normalized = Column(String(255), nullable=True, index=True)
    description = Column(String(500), nullable=True)
    transaction_date = Column(Date, nullable=True, index=True)
    gross_amount_cents = Column(Integer, nullable=False)
    retainage_amount_cents = Column(Integer, default=0)  # Amount held back
    allocation_method = Column(String(20), default='direct')  # direct, split_50_50
    zone = Column(String(10), nullable=True, index=True)  # Derived from budget
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    budget = relationship("BudgetEntity", back_populates="direct_costs")

    @property
    def payable_amount_cents(self) -> int:
        """Net payment = Gross - Retainage held."""
        return self.gross_amount_cents - self.retainage_amount_cents


# =============================================================================
# Training Round Entity (The "Brain" Versioning)
# =============================================================================

class TrainingRound(Base):
    """
    Captures a version of the calculation/ML engine.
    Created each time the system "learns" from user feedback.
    """
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)  # External UUID
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True, index=True)
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    trigger_type = Column(String(50), default='manual')  # manual, nightly, file_change, user_feedback
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    completed_at = Column(DateTime, nullable=True)

    # Performance metrics
    linkage_score = Column(Float, nullable=True)  # % of costs successfully linked to schedule
    mapping_accuracy = Column(Float, nullable=True)  # % of mappings with high confidence
    budget_coverage = Column(Float, nullable=True)  # % of budgets linked to GMP
    cost_coverage = Column(Float, nullable=True)  # % of costs linked to budgets

    # Model metadata
    model_version = Column(String(50), nullable=True)
    model_params = Column(Text, nullable=True)  # JSON of model parameters
    training_notes = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Comparison with previous round
    previous_round_id = Column(Integer, nullable=True)
    eac_change_cents = Column(Integer, nullable=True)  # Change in total EAC from previous
    eac_change_pct = Column(Float, nullable=True)  # % change in EAC

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="training_rounds")
    forecast_snapshots = relationship("TrainingForecastSnapshot", back_populates="training_round")


class TrainingForecastSnapshot(Base):
    """
    Forecast curve points for a specific training round.
    Stores predicted cumulative costs over time, by zone.
    """
    __tablename__ = "training_forecast_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    training_round_id = Column(Integer, ForeignKey('training_rounds.id'), nullable=False, index=True)
    period_date = Column(Date, nullable=False, index=True)
    predicted_cumulative_cost_cents = Column(Integer, nullable=False)
    actual_cumulative_cost_cents = Column(Integer, nullable=True)  # If available at time
    zone = Column(String(10), nullable=False, index=True)  # EAST, WEST, SHARED
    confidence_lower_cents = Column(Integer, nullable=True)  # Lower bound
    confidence_upper_cents = Column(Integer, nullable=True)  # Upper bound
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    training_round = relationship("TrainingRound", back_populates="forecast_snapshots")

    __table_args__ = (
        UniqueConstraint('training_round_id', 'period_date', 'zone', name='uq_training_period_zone'),
    )


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

    P6 Date Format: "DD-Mon-YY[ A]" where " A" suffix indicates actual date
    - Both dates have " A": activity is COMPLETE (progress = 1.0)
    - Only start has " A": activity is IN_PROGRESS (progress = elapsed/duration)
    - Neither date has " A": activity is NOT_STARTED (progress = 0.0)
    """
    __tablename__ = 'schedule_activities'

    id = Column(Integer, primary_key=True, index=True)
    row_number = Column(Integer)
    task_name = Column(String(500), nullable=False)
    source_uid = Column(String(100), unique=True, nullable=True)  # P6 GUID
    activity_id = Column(String(50), index=True)
    wbs = Column(String(100), index=True)
    pct_complete = Column(Integer, default=0)  # 0-100 (explicit if provided)

    # Schedule dates for time-based forecasting
    start_date = Column(Date, nullable=True)  # Current/actual start
    finish_date = Column(Date, nullable=True)  # Current/actual finish
    planned_start = Column(Date, nullable=True)  # Baseline start
    planned_finish = Column(Date, nullable=True)  # Baseline finish
    duration_days = Column(Integer, nullable=True)

    # P6-specific: Track whether dates are actuals (had " A" suffix)
    start_is_actual = Column(Boolean, default=False)  # Start date had " A" suffix
    finish_is_actual = Column(Boolean, default=False)  # Finish date had " A" suffix

    # P6 derived progress state (computed from date suffixes)
    is_complete = Column(Boolean, default=False)  # Both dates actual
    is_in_progress = Column(Boolean, default=False)  # Only start is actual
    progress_pct = Column(Float, default=0.0)  # 0.0-1.0 computed progress

    # Critical path weighting
    total_float = Column(Integer, nullable=True)  # 0 = critical path
    is_critical = Column(Boolean, default=False)  # total_float == 0

    # GMP mapping source tracking
    mapping_source = Column(String(30), default='manual')  # manual, prefix_match, keyword_match, fuzzy_match
    mapping_confidence = Column(Float, default=1.0)  # 0.0-1.0

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
