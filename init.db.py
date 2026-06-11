from app.core.database import SQL_ENGINE

from app.models.base import Base

def main() -> None:
    Base.metadata.create_all(bind=SQL_ENGINE)

if __name__ == '__main__':
    main()
