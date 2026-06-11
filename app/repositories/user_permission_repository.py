from sqlalchemy.orm import Session as DbSession

from app.models.user_permission import UserPermission


class UserPermissionRepository:
    def __init__(self, db: DbSession):
        self.db = db

    def replace_for_user(self, user_id: str, permissions: list[str]) -> None:
        self.db.query(UserPermission).where(UserPermission.user_id == user_id).delete()
        for permission_code in sorted(set(permissions)):
            normalized_permission = permission_code.strip()
            if not normalized_permission:
                continue
            self.db.add(
                UserPermission(
                    user_id=user_id,
                    permission_code=normalized_permission,
                )
            )
        self.db.flush()
