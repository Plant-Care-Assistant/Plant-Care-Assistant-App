from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session, create_engine

from app.settings import settings

engine = create_engine(settings.db_url)


def start_session() -> Generator[Session]:
    with Session(engine) as session:
        yield session
        session.commit()


DbDepends = Annotated[Session, Depends(start_session)]
