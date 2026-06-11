from datetime import date

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session as DbSession

from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User


class UsageRepository:
    def __init__(self, db: DbSession):
        self.db = db

    def daily_usage_for_tenant(self, tenant_id: str | None, start_date: date, end_date: date):
        usage_date = func.date(Message.created_at).label("usage_date")
        return self.db.execute(
            select(
                usage_date,
                func.count(distinct(Conversation.owner_id)).label("active_users"),
                func.coalesce(func.sum(Message.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(Message.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(Message.total_tokens), 0).label("total_tokens"),
            )
            .join(Conversation, Message.conversation_id == Conversation.id)
            .join(User, Conversation.owner_id == User.id)
            .where(
                User.tenant_id == tenant_id,
                func.date(Message.created_at) >= start_date.isoformat(),
                func.date(Message.created_at) <= end_date.isoformat(),
            )
            .group_by(usage_date)
            .order_by(usage_date)
        ).all()
