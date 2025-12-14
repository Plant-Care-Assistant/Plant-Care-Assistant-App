
from sqlmodel import select

from app.db import DbDepends
from app.models.base import User


def read_users(session: DbDepends) -> list[User]:
    statement = select(User)
    return list(session.exec(statement).all())


def read_user_id(session: DbDepends, user_id: int) -> User | None:
    statement = select(User).where(User.id == user_id)
    return session.exec(statement).one_or_none()


def read_user_login(session: DbDepends, login: str) -> User | None:
    statement = select(User).where(User.login == login)
    return session.exec(statement).one_or_none()
