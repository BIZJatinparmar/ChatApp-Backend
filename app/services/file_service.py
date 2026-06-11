import os
from uuid import uuid4

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FileService:
    @staticmethod
    def build_faiss_index(pdf_path: str, index_dir: str) -> None:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()

        if os.path.exists(index_dir):
            vector_db = FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            vector_db.add_documents(chunks)
        else:
            vector_db = FAISS.from_documents(chunks, embeddings)

        vector_db.save_local(index_dir)

    async def upload_file(self, file: UploadFile) -> dict[str, str | int | None]:
        content = await file.read()
        doc_id = str(uuid4())
        path = os.path.join(os.getcwd(), "../uploaded_files", f"{doc_id}_{file.filename}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as target:
            target.write(content)

        self.build_faiss_index(path, os.path.join(os.getcwd(), "../faiss_indices"))
        return {"filename": file.filename, "content_type": file.content_type, "doc_id": doc_id}
