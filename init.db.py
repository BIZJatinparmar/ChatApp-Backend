from app.core.database import SQL_ENGINE

import app.models.budget_request  # noqa: F401
import app.models.conversation  # noqa: F401
import app.models.document_model  # noqa: F401
import app.models.message  # noqa: F401
import app.models.rag_retrieval_event  # noqa: F401
import app.models.user  # noqa: F401
import app.models.user_permission  # noqa: F401
from app.models.base import Base

def main() -> None:
    Base.metadata.create_all(bind=SQL_ENGINE)

if __name__ == '__main__':
    main()
