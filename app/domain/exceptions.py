"""
Domain Exceptions for Cost Management Hierarchy.

Custom exceptions enforcing business rules:
- GMP immutability
- Budget ceiling constraints
- Mapping integrity
- Temporal consistency
"""


class DomainError(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, code: str = "DOMAIN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


# =============================================================================
# GMP Exceptions
# =============================================================================

class ImmutableFieldError(DomainError):
    """Raised when attempting to modify an immutable field."""

    def __init__(self, field_name: str, entity_type: str = "GMP"):
        message = (
            f"{entity_type} {field_name} cannot be modified after creation. "
            f"Use Change Orders for budget modifications."
        )
        super().__init__(message, code="IMMUTABLE_FIELD")
        self.field_name = field_name
        self.entity_type = entity_type


class GMPNotFoundError(DomainError):
    """Raised when a GMP entity cannot be found."""

    def __init__(self, gmp_id: str):
        message = f"GMP with id '{gmp_id}' not found"
        super().__init__(message, code="GMP_NOT_FOUND")
        self.gmp_id = gmp_id


class DuplicateGMPError(DomainError):
    """Raised when attempting to create a duplicate GMP for a division."""

    def __init__(self, division: str, project_id: str):
        message = f"GMP for division '{division}' already exists in project '{project_id}'"
        super().__init__(message, code="DUPLICATE_GMP")
        self.division = division
        self.project_id = project_id


# =============================================================================
# Budget Exceptions
# =============================================================================

class GMPCeilingExceededError(DomainError):
    """Raised when budget allocation would exceed GMP ceiling."""

    def __init__(
        self,
        total_budgeted: int,
        gmp_amount: int,
        available: int
    ):
        message = (
            f"Total budgets ({total_budgeted:,} cents) would exceed "
            f"GMP ceiling ({gmp_amount:,} cents). "
            f"Available: {available:,} cents"
        )
        super().__init__(message, code="GMP_CEILING_EXCEEDED")
        self.total_budgeted = total_budgeted
        self.gmp_amount = gmp_amount
        self.available = available


class BudgetUnderflowError(DomainError):
    """Raised when attempting to reduce budget below actual spent."""

    def __init__(self, budget_amount: int, actual_cost: int):
        message = (
            f"Cannot set budget ({budget_amount:,} cents) "
            f"below actual cost ({actual_cost:,} cents)"
        )
        super().__init__(message, code="BUDGET_UNDERFLOW")
        self.budget_amount = budget_amount
        self.actual_cost = actual_cost


class BudgetNotFoundError(DomainError):
    """Raised when a budget entity cannot be found."""

    def __init__(self, budget_id: str):
        message = f"Budget with id '{budget_id}' not found"
        super().__init__(message, code="BUDGET_NOT_FOUND")
        self.budget_id = budget_id


class BudgetHasMappedCostsError(DomainError):
    """Raised when attempting to delete a budget with mapped direct costs."""

    def __init__(self, budget_id: str, cost_count: int):
        message = (
            f"Cannot delete budget '{budget_id}': "
            f"{cost_count} direct costs are mapped to it. "
            f"Reassign or delete costs first."
        )
        super().__init__(message, code="BUDGET_HAS_MAPPED_COSTS")
        self.budget_id = budget_id
        self.cost_count = cost_count


# =============================================================================
# Direct Cost Exceptions
# =============================================================================

class DirectCostNotFoundError(DomainError):
    """Raised when a direct cost entity cannot be found."""

    def __init__(self, cost_id: str):
        message = f"Direct cost with id '{cost_id}' not found"
        super().__init__(message, code="DIRECT_COST_NOT_FOUND")
        self.cost_id = cost_id


class InvalidMappingError(DomainError):
    """Raised when a cost mapping is invalid."""

    def __init__(self, cost_id: str, budget_id: str, reason: str):
        message = (
            f"Invalid mapping: cost '{cost_id}' cannot be mapped to "
            f"budget '{budget_id}'. Reason: {reason}"
        )
        super().__init__(message, code="INVALID_MAPPING")
        self.cost_id = cost_id
        self.budget_id = budget_id
        self.reason = reason


class InvalidTransactionDateError(DomainError):
    """Raised when a transaction date is invalid."""

    def __init__(self, cost_id: str):
        message = f"Direct cost '{cost_id}' has an invalid transaction date"
        super().__init__(message, code="INVALID_TRANSACTION_DATE")
        self.cost_id = cost_id


# =============================================================================
# Schedule Exceptions
# =============================================================================

class ScheduleNotFoundError(DomainError):
    """Raised when a schedule cannot be found."""

    def __init__(self, schedule_id: str):
        message = f"Schedule with id '{schedule_id}' not found"
        super().__init__(message, code="SCHEDULE_NOT_FOUND")
        self.schedule_id = schedule_id


class ScheduleActivityNotFoundError(DomainError):
    """Raised when a schedule activity cannot be found."""

    def __init__(self, activity_id: str):
        message = f"Schedule activity with id '{activity_id}' not found"
        super().__init__(message, code="SCHEDULE_ACTIVITY_NOT_FOUND")
        self.activity_id = activity_id


class InvalidScheduleDateRangeError(DomainError):
    """Raised when schedule dates are invalid."""

    def __init__(self, start_date, end_date):
        message = (
            f"Schedule end date ({end_date}) must be after "
            f"start date ({start_date})"
        )
        super().__init__(message, code="INVALID_SCHEDULE_DATE_RANGE")
        self.start_date = start_date
        self.end_date = end_date


class ActivityOutOfScheduleBoundsError(DomainError):
    """Raised when an activity's dates fall outside the schedule bounds."""

    def __init__(self, activity_id: str, schedule_id: str):
        message = (
            f"Activity '{activity_id}' dates must be within "
            f"schedule '{schedule_id}' bounds"
        )
        super().__init__(message, code="ACTIVITY_OUT_OF_BOUNDS")
        self.activity_id = activity_id
        self.schedule_id = schedule_id


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(DomainError):
    """Raised when data validation fails."""

    def __init__(self, field: str, message: str):
        super().__init__(f"Validation failed for '{field}': {message}", code="VALIDATION_ERROR")
        self.field = field


class ConcurrencyError(DomainError):
    """Raised when optimistic locking fails (version mismatch)."""

    def __init__(self, entity_type: str, entity_id: str):
        message = (
            f"Concurrent modification detected for {entity_type} '{entity_id}'. "
            f"Please refresh and try again."
        )
        super().__init__(message, code="CONCURRENCY_ERROR")
        self.entity_type = entity_type
        self.entity_id = entity_id


# =============================================================================
# Aggregation Exceptions
# =============================================================================

class ReconciliationError(DomainError):
    """Raised when reconciliation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="RECONCILIATION_ERROR")
        self.details = details or {}


class InvariantViolationError(DomainError):
    """Raised when a mathematical invariant is violated."""

    def __init__(self, invariant_name: str, expected: str, actual: str):
        message = (
            f"Invariant '{invariant_name}' violated. "
            f"Expected: {expected}, Actual: {actual}"
        )
        super().__init__(message, code="INVARIANT_VIOLATION")
        self.invariant_name = invariant_name
        self.expected = expected
        self.actual = actual
