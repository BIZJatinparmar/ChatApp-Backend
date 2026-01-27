from db import SQL_ENGINE

from models.Base import Base

def main() -> None:
    Base.metadata.create_all(bind=SQL_ENGINE)

if __name__ == '__main__':
    main()