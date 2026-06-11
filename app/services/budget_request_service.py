from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from app.models.budget_request import BudgetRequest
from app.models.user import User
from app.repositories.budget_request_repository import BudgetRequestRepository
from app.schemas.budget_request import (
    BudgetRequestAdminUpdateIn,
    BudgetRequestCreateIn,
    BudgetRequestOut,
    BudgetRequestStatus,
)


def to_budget_request_out(request: BudgetRequest) -> BudgetRequestOut:
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


class BudgetRequestService:
    def __init__(self, db: DbSession):
        self.db = db
        self.requests = BudgetRequestRepository(db)

    def create_request(self, payload: BudgetRequestCreateIn, user: User) -> BudgetRequestOut:
        request = self.requests.create(
            BudgetRequest(
                user_id=user.id,
                requested_tokens=payload.requested_tokens,
                note=payload.note.strip() if payload.note else None,
            )
        )
        self.db.commit()
        self.db.refresh(request)
        return to_budget_request_out(request)

    def list_for_user(self, user: User) -> list[BudgetRequestOut]:
        return [to_budget_request_out(request) for request in self.requests.list_for_user(user.id)]

    def list_for_admin(self, admin_user: User) -> list[BudgetRequestOut]:
        return [to_budget_request_out(request) for request in self.requests.list_for_tenant(admin_user.tenant_id)]

    def decide_request(
        self,
        request_id: str,
        payload: BudgetRequestAdminUpdateIn,
        admin_user: User,
    ) -> BudgetRequestOut:
        request = self.requests.get_by_id(request_id)
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

        self.db.commit()
        self.db.refresh(request)
        return to_budget_request_out(request)
