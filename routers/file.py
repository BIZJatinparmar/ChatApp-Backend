from fastapi import APIRouter, UploadFile, File
from uuid import uuid4
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import Depends

from deps.auth import require_permissions
from models.user import User

router = APIRouter(prefix="/files", tags=["files"])


def build_faiss_index(pdf_path: str, index_dir: str) -> None:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = os.path.basename(pdf_path)  # type: ignore

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_dir):
        vectordb = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectordb.add_documents(chunks)
    else:
        vectordb = FAISS.from_documents(chunks, embeddings)

    vectordb.save_local(index_dir)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: User = Depends(require_permissions("files:upload")),
) -> dict[str, str | int | None]:
    content = await file.read()
    doc_id = str(uuid4())
    path = os.path.join(os.getcwd(), "../uploaded_files", f"{doc_id}_{file.filename}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    
    build_faiss_index(path, os.path.join(os.getcwd(), "../faiss_indices"))
    return {"filename": file.filename, "content_type": file.content_type, "doc_id": doc_id}
