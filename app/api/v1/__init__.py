"""
API v1 - REST endpoints for cost management entities.

Implements the API contracts from the specification:
- Auth endpoints (register, login)
- GMP endpoints (create, read, list)
- Budget endpoints (CRUD)
- Direct Cost endpoints (CRUD, bulk mapping)
- Schedule endpoints (read, progress updates)
- Migration endpoints (Phase 2 multi-project)
"""
from fastapi import APIRouter

from .auth import router as auth_router
from .gmp import router as gmp_router
from .budgets import router as budgets_router
from .direct_costs import router as direct_costs_router
from .reconciliation import router as reconciliation_router
from .migration import router as migration_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(gmp_router, prefix="/gmp", tags=["GMP"])
api_router.include_router(budgets_router, prefix="/budgets", tags=["Budgets"])
api_router.include_router(direct_costs_router, prefix="/direct-costs", tags=["Direct Costs"])
api_router.include_router(reconciliation_router, prefix="/reconciliation", tags=["Reconciliation"])
api_router.include_router(migration_router, prefix="/migration", tags=["Migration"])
