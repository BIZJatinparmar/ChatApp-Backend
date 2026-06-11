from fastapi import APIRouter, UploadFile, File
from fastapi import Depends

from app.dependencies.auth import require_permissions
from app.models.user import User
from app.services.file_service import FileService

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: User = Depends(require_permissions("files:upload")),
) -> dict[str, str | int | None]:
    return await FileService().upload_file(file)
