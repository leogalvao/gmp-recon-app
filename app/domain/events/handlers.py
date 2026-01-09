"""
Domain Event Handlers for Cost Management Hierarchy.

Implements SQLAlchemy event listeners for:
- GMP immutability enforcement
- Budget ceiling validation
- Direct cost cascading updates
- Audit trail recording

These handlers ensure business rules are enforced at the ORM level.
"""
import json
from datetime import datetime
from typing import Optional, Set
from sqlalchemy import event, inspect
from sqlalchemy.orm import Session

from app.models import (
    GMP,
    BudgetEntity,
    DirectCostEntity,
    ChangeOrder,
    MappingAudit,
    ChangeOrderStatus,
)
from app.domain.exceptions import (
    ImmutableFieldError,
    GMPCeilingExceededError,
    BudgetUnderflowError,
)


# =============================================================================
# GMP Event Handlers - Immutability Enforcement
# =============================================================================

# Track which GMP instances are newly created vs loaded from DB
_NEW_GMP_INSTANCES: Set[int] = set()


@event.listens_for(GMP, 'init')
def gmp_on_init(target, args, kwargs):
    """Mark new GMP instances before they're persisted."""
    # Use object id to track new instances
    _NEW_GMP_INSTANCES.add(id(target))


@event.listens_for(GMP, 'load')
def gmp_on_load(target, context):
    """Clear new instance marker when GMP is loaded from database."""
    # Loaded instances are not new
    _NEW_GMP_INSTANCES.discard(id(target))


@event.listens_for(GMP, 'after_insert')
def gmp_after_insert(mapper, connection, target):
    """Mark instance as persisted (no longer new)."""
    _NEW_GMP_INSTANCES.discard(id(target))


@event.listens_for(GMP, 'before_update')
def gmp_before_update(mapper, connection, target):
    """
    Enforce GMP amount immutability.

    The original_amount_cents field CANNOT be modified after creation.
    Only description can be updated.
    """
    # Get the history of changes for this instance
    state = inspect(target)

    # Check if original_amount_cents was modified
    history = state.attrs.original_amount_cents.history

    if history.has_changes():
        old_value = history.deleted[0] if history.deleted else None
        new_value = history.added[0] if history.added else target.original_amount_cents

        if old_value is not None and old_value != new_value:
            raise ImmutableFieldError(
                field_name='original_amount_cents',
                entity_type='GMP'
            )


def validate_gmp_amount_positive(amount_cents: int) -> bool:
    """Validate GMP amount is positive."""
    if amount_cents <= 0:
        raise ValueError("GMP amount must be positive")
    return True


# =============================================================================
# Budget Event Handlers - Ceiling Constraint Validation
# =============================================================================

def get_gmp_ceiling(session: Session, gmp_id: int) -> int:
    """Get the authorized GMP ceiling (original + approved COs)."""
    gmp = session.query(GMP).filter(GMP.id == gmp_id).first()
    if not gmp:
        return 0
    return gmp.authorized_amount_cents


def get_total_budgeted(session: Session, gmp_id: int, exclude_budget_id: Optional[int] = None) -> int:
    """Get sum of all budgets for a GMP, optionally excluding one budget."""
    query = session.query(BudgetEntity).filter(BudgetEntity.gmp_id == gmp_id)
    if exclude_budget_id:
        query = query.filter(BudgetEntity.id != exclude_budget_id)

    return sum(b.current_budget_cents or 0 for b in query.all())


def get_actual_cost_for_budget(session: Session, budget_id: int) -> int:
    """Get sum of all direct costs mapped to a budget."""
    costs = session.query(DirectCostEntity).filter(
        DirectCostEntity.mapped_budget_id == budget_id
    ).all()
    return sum(c.gross_amount_cents for c in costs)


def validate_gmp_ceiling(session: Session, gmp_id: int, new_budget_amount: int,
                         exclude_budget_id: Optional[int] = None) -> bool:
    """
    Validate that adding/updating a budget won't exceed GMP ceiling.

    Args:
        session: Database session
        gmp_id: Parent GMP identifier
        new_budget_amount: Amount to validate (in cents)
        exclude_budget_id: Budget ID to exclude (for updates)

    Returns:
        True if valid

    Raises:
        GMPCeilingExceededError if ceiling would be exceeded
    """
    gmp_ceiling = get_gmp_ceiling(session, gmp_id)
    existing_total = get_total_budgeted(session, gmp_id, exclude_budget_id)
    new_total = existing_total + new_budget_amount

    if new_total > gmp_ceiling:
        raise GMPCeilingExceededError(
            total_budgeted=new_total,
            gmp_amount=gmp_ceiling,
            available=gmp_ceiling - existing_total
        )

    return True


def validate_budget_not_below_actual(session: Session, budget_id: int,
                                     new_amount: int) -> bool:
    """
    Validate that budget is not reduced below actual spent.

    Args:
        session: Database session
        budget_id: Budget identifier
        new_amount: New budget amount (in cents)

    Returns:
        True if valid

    Raises:
        BudgetUnderflowError if budget would be below actual
    """
    actual_cost = get_actual_cost_for_budget(session, budget_id)

    if new_amount < actual_cost:
        raise BudgetUnderflowError(
            budget_amount=new_amount,
            actual_cost=actual_cost
        )

    return True


# =============================================================================
# Direct Cost Event Handlers - Cascading Updates
# =============================================================================

def derive_gmp_id_from_budget(session: Session, budget_id: Optional[int]) -> Optional[int]:
    """Get the GMP ID from a budget's parent relationship."""
    if not budget_id:
        return None
    budget = session.query(BudgetEntity).filter(BudgetEntity.id == budget_id).first()
    return budget.gmp_id if budget else None


def derive_zone_from_budget(session: Session, budget_id: Optional[int]) -> Optional[str]:
    """Get the zone from a budget's assignment."""
    if not budget_id:
        return None
    budget = session.query(BudgetEntity).filter(BudgetEntity.id == budget_id).first()
    return budget.zone if budget else None


# =============================================================================
# Audit Trail Recording
# =============================================================================

def record_audit_log(session: Session, table_name: str, record_id: int,
                     action: str, old_value: Optional[dict] = None,
                     new_value: Optional[dict] = None, user: str = "system"):
    """
    Record an audit log entry for a mapping/entity change.

    Args:
        session: Database session
        table_name: Name of the table being modified
        record_id: ID of the record being modified
        action: Type of action (create, update, delete)
        old_value: Previous values (for update/delete)
        new_value: New values (for create/update)
        user: User performing the action
    """
    audit = MappingAudit(
        table_name=table_name,
        record_id=record_id,
        action=action,
        old_value=json.dumps(old_value) if old_value else None,
        new_value=json.dumps(new_value) if new_value else None,
        user=user,
        timestamp=datetime.utcnow()
    )
    session.add(audit)


def get_entity_dict(entity) -> dict:
    """Convert SQLAlchemy entity to dictionary for audit logging."""
    result = {}
    for column in entity.__table__.columns:
        value = getattr(entity, column.name)
        if hasattr(value, 'isoformat'):
            value = value.isoformat()
        elif hasattr(value, 'value'):  # Enum
            value = value.value
        result[column.name] = value
    return result


# =============================================================================
# Budget Audit Listeners
# =============================================================================

@event.listens_for(BudgetEntity, 'after_insert')
def budget_after_insert(mapper, connection, target):
    """Record audit log for budget creation."""
    # Note: We can't use session here directly in the after_insert
    # This is handled at the service layer instead
    pass


@event.listens_for(BudgetEntity, 'before_update')
def budget_before_update(mapper, connection, target):
    """Track changes to budget for audit logging."""
    state = inspect(target)

    # Store old values for later audit recording
    changes = {}
    for attr in ['current_budget_cents', 'description', 'committed_cents', 'zone']:
        history = state.attrs[attr].history
        if history.has_changes():
            changes[attr] = {
                'old': history.deleted[0] if history.deleted else None,
                'new': history.added[0] if history.added else getattr(target, attr)
            }

    # Store on the instance for service layer to use
    target._pending_changes = changes


# =============================================================================
# Direct Cost Cascade Tracking
# =============================================================================

@event.listens_for(DirectCostEntity, 'before_update')
def direct_cost_before_update(mapper, connection, target):
    """Track mapping changes for cascade updates."""
    state = inspect(target)

    # Track old mapping for cascade recalculation
    history = state.attrs.mapped_budget_id.history
    if history.has_changes():
        target._old_budget_id = history.deleted[0] if history.deleted else None
        target._new_budget_id = history.added[0] if history.added else target.mapped_budget_id


# =============================================================================
# Registration Helper
# =============================================================================

def register_all_event_handlers():
    """
    Register all event handlers.

    This function is called during application startup to ensure
    all SQLAlchemy event listeners are properly registered.
    """
    # Event listeners are registered via decorators above
    # This function serves as documentation and could be extended
    # for dynamic registration if needed
    pass
