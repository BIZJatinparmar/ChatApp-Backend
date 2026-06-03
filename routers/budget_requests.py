from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from db import get_db
from deps.auth import get_current_user, require_role
from models.budget_request import BudgetRequest
from models.user import User
from schemas.budget_request import (
    BudgetRequestAdminUpdateIn,
    BudgetRequestCreateIn,
    BudgetRequestOut,
    BudgetRequestStatus,
)
from schemas.user import AppRole

router = APIRouter(tags=["budget-requests"])


def _to_budget_request_out(request: BudgetRequest) -> BudgetRequestOut:
    return BudgetRequestOut(
        id=request.id,
        user_id=request.user_id,
        user_email=request.user.email if request.user else None,
        requested_tokens=request.requested_tokens,
        status=BudgetRequestStatus(request.status),
        note=request.note,
        admin_note=request.admin_note,
        decided_by_user_id=request.decided_by_user_id,
        decided_at=request.decided_at,
        created_at=request.created_at,
        updated_at=request.updated_at,
    )


@router.post("/budget-requests", response_model=BudgetRequestOut, status_code=201)
def create_budget_request(
    payload: BudgetRequestCreateIn,
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    request = BudgetRequest(
        user_id=user.id,
        requested_tokens=payload.requested_tokens,
        note=payload.note.strip() if payload.note else None,
    )
    db.add(request)
    db.commit()
    db.refresh(request)
    return _to_budget_request_out(request)


@router.get("/budget-requests/me", response_model=list[BudgetRequestOut])
def list_my_budget_requests(
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_db),
):
    requests = db.scalars(
        select(BudgetRequest)
        .where(BudgetRequest.user_id == user.id)
        .order_by(BudgetRequest.created_at.desc())
    ).all()
    return [_to_budget_request_out(request) for request in requests]


@router.get("/admin/budget-requests", response_model=list[BudgetRequestOut])
def list_admin_budget_requests(
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    requests = db.scalars(
        select(BudgetRequest)
        .join(User, BudgetRequest.user_id == User.id)
        .where(User.tenant_id == admin_user.tenant_id)
        .order_by(BudgetRequest.created_at.desc())
    ).all()
    return [_to_budget_request_out(request) for request in requests]


@router.patch("/admin/budget-requests/{request_id}", response_model=BudgetRequestOut)
def decide_budget_request(
    request_id: str,
    payload: BudgetRequestAdminUpdateIn,
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    request = db.get(BudgetRequest, request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Budget request not found")
    if not request.user or request.user.tenant_id != admin_user.tenant_id:
        raise HTTPException(status_code=403, detail="Budget request belongs to another tenant")
    if request.status != BudgetRequestStatus.pending.value:
        raise HTTPException(status_code=409, detail="Budget request is already decided")
    if payload.status == BudgetRequestStatus.approved and payload.approved_tokens is None:
        raise HTTPException(status_code=400, detail="approved_tokens is required when approving")

    request.status = payload.status.value
    request.admin_note = payload.admin_note.strip() if payload.admin_note else None
    request.decided_by_user_id = admin_user.id
    request.decided_at = datetime.now(timezone.utc)

    if payload.status == BudgetRequestStatus.approved:
        request.user.token_budget += int(payload.approved_tokens or 0)

    db.commit()
    db.refresh(request)
    return _to_budget_request_out(request)
