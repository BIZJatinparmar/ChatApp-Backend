from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session
from app.core.database import get_db
from app.dependencies.auth import require_permissions
from app.models.user import User
from app.schemas.conversation import ConversationCreateRequest
from app.services.conversation_service import ConversationService


router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.get("/")
def conversations(
    db: Session = Depends(get_db),
    user: User = Depends(require_permissions("conversation:read")),
):
    return ConversationService(db).list_for_user(user)


@router.get("/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_permissions("conversation:read")),
):
    return ConversationService(db).list_messages(conversation_id, user)


@router.post("/")
async def create_conversation(
    body: ConversationCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permissions("conversation:write")),
):
    return ConversationService(db).create_conversation(body, user)
