from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from app.core.database import get_db
from app.dependencies.auth import get_current_user, require_role
from app.models.user import User
from app.schemas.budget_request import (
    BudgetRequestAdminUpdateIn,
    BudgetRequestCreateIn,
    BudgetRequestOut,
)
from app.schemas.user import AppRole
from app.services.budget_request_service import BudgetRequestService

router = APIRouter(tags=["budget-requests"])


@router.post("/budget-requests", response_model=BudgetRequestOut, status_code=201)
def create_budget_request(
    payload: BudgetRequestCreateIn,
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    return BudgetRequestService(db).create_request(payload, user)


@router.get("/budget-requests/me", response_model=list[BudgetRequestOut])
def list_my_budget_requests(
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    return BudgetRequestService(db).list_for_user(user)


@router.get("/admin/budget-requests", response_model=list[BudgetRequestOut])
def list_admin_budget_requests(
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    return BudgetRequestService(db).list_for_admin(admin_user)


@router.patch("/admin/budget-requests/{request_id}", response_model=BudgetRequestOut)
def decide_budget_request(
    request_id: str,
    payload: BudgetRequestAdminUpdateIn,
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    return BudgetRequestService(db).decide_request(request_id, payload, admin_user)
