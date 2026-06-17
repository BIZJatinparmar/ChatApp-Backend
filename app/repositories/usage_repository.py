from datetime import date

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session as DbSession

from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User


UsageTotals = dict[str, int]


def _empty_usage() -> UsageTotals:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class UsageRepository:
    def __init__(self, db: DbSession):
        self.db = db

    def usage_for_user(self, user_id: str) -> UsageTotals:
        return self.usage_for_users([user_id]).get(user_id, _empty_usage())

    def usage_for_users(self, user_ids: list[str]) -> dict[str, UsageTotals]:
        usage_by_user = {user_id: _empty_usage() for user_id in user_ids}
        if not user_ids:
            return usage_by_user

        rows = self.db.execute(
            select(
                Conversation.owner_id,
                func.coalesce(func.sum(Message.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(Message.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(Message.total_tokens), 0).label("total_tokens"),
            )
            .join(Message, Message.conversation_id == Conversation.id)
            .where(Conversation.owner_id.in_(user_ids))
            .group_by(Conversation.owner_id)
        ).all()

        for row in rows:
            usage_by_user[row.owner_id] = {
                "input_tokens": int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "total_tokens": int(row.total_tokens or 0),
            }
        return usage_by_user

    def usage_for_conversation(self, conversation_id: str) -> UsageTotals:
        return self.usage_for_conversations([conversation_id]).get(
            conversation_id,
            _empty_usage(),
        )

    def usage_for_conversations(self, conversation_ids: list[str]) -> dict[str, UsageTotals]:
        usage_by_conversation = {
            conversation_id: _empty_usage() for conversation_id in conversation_ids
        }
        if not conversation_ids:
            return usage_by_conversation

        rows = self.db.execute(
            select(
                Message.conversation_id,
                func.coalesce(func.sum(Message.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(Message.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(Message.total_tokens), 0).label("total_tokens"),
            )
            .where(Message.conversation_id.in_(conversation_ids))
            .group_by(Message.conversation_id)
        ).all()

        for row in rows:
            usage_by_conversation[row.conversation_id] = {
                "input_tokens": int(row.input_tokens or 0),
                "output_tokens": int(row.output_tokens or 0),
                "total_tokens": int(row.total_tokens or 0),
            }
        return usage_by_conversation

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
