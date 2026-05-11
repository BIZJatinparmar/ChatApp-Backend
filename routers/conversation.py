import select

from fastapi import APIRouter, Depends

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session
from db import get_db
from models.conversation import Conversation
from pydantic import BaseModel


class ConversationResponse(BaseModel):
    id: int
    owner_id: int
    title: str

    model_config = {"from_attributes": True}


router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.get("/", response_model=list[ConversationResponse])
async def get_conversations(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation).where(Conversation.owner_id == "1")
    )
    return result.scalars().all()
