"""
Database models and SQLAlchemy setup for GMP Reconciliation App.
All monetary values stored as integer cents to avoid float drift.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
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
