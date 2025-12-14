from fastapi import Depends
from app.settings import settings
from sqlmodel import create_engine, Session
from typing import Annotated

engine = create_engine(settings.db_url.unicode_string())

def start_session():
    with Session(engine) as session:
        yield session
        session.commit()

DbDepends = Annotated[Session, Depends(start_session)]
