from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.models.budget_request import BudgetRequest
from app.models.user import User


class BudgetRequestRepository:
    def __init__(self, db: DbSession):
        self.db = db

    def create(self, request: BudgetRequest) -> BudgetRequest:
        self.db.add(request)
        self.db.flush()
        self.db.refresh(request)
        return request

    def get_by_id(self, request_id: str) -> BudgetRequest | None:
        return self.db.get(BudgetRequest, request_id)

    def list_for_user(self, user_id: str) -> list[BudgetRequest]:
        return list(
            self.db.scalars(
                select(BudgetRequest)
                .where(BudgetRequest.user_id == user_id)
                .order_by(BudgetRequest.created_at.desc())
            ).all()
        )

    def list_for_tenant(self, tenant_id: str | None) -> list[BudgetRequest]:
        return list(
            self.db.scalars(
                select(BudgetRequest)
                .join(User, BudgetRequest.user_id == User.id)
                .where(User.tenant_id == tenant_id)
                .order_by(BudgetRequest.created_at.desc())
            ).all()
        )
