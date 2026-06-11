from datetime import date, timedelta

from sqlalchemy.orm import Session as DbSession

from app.models.user import User
from app.repositories.usage_repository import UsageRepository
from app.schemas.admin_usage import DailyUsageOut


class AdminUsageService:
    def __init__(self, db: DbSession):
        self.usage = UsageRepository(db)

    def get_daily_usage(self, days: int, admin_user: User) -> list[DailyUsageOut]:
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        rows = self.usage.daily_usage_for_tenant(admin_user.tenant_id, start_date, end_date)

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
