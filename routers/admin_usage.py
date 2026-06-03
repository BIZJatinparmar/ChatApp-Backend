from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session as DbSession

from db import get_db
from deps.auth import require_role
from models.conversation import Conversation
from models.message import Message
from models.user import User
from schemas.admin_usage import DailyUsageOut
from schemas.user import AppRole

router = APIRouter(prefix="/admin/usage", tags=["admin"])


@router.get("/daily", response_model=list[DailyUsageOut])
def get_daily_usage(
    days: int = Query(default=30, ge=1, le=90),
    admin_user: User = Depends(require_role(AppRole.admin.value)),
    db: DbSession = Depends(get_db),
):
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)

    usage_date = func.date(Message.created_at).label("usage_date")
    rows = db.execute(
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
            User.tenant_id == admin_user.tenant_id,
            func.date(Message.created_at) >= start_date.isoformat(),
            func.date(Message.created_at) <= end_date.isoformat(),
        )
        .group_by(usage_date)
        .order_by(usage_date)
    ).all()

    usage_by_date = {
        str(row.usage_date): DailyUsageOut(
            date=str(row.usage_date),
            active_users=int(row.active_users or 0),
            input_tokens=int(row.input_tokens or 0),
            output_tokens=int(row.output_tokens or 0),
            total_tokens=int(row.total_tokens or 0),
        )
        for row in rows
    }

    return [
        usage_by_date.get(
            current_date.isoformat(),
            DailyUsageOut(date=current_date.isoformat()),
        )
        for current_date in (
            start_date + timedelta(days=offset) for offset in range(days)
        )
    ]
