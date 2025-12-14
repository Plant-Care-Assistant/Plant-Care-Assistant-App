from uuid import UUID

from app.routers import users
from sqlmodel import select
from app.db import DbDepends
from app.models.base import User


def read_users(session: DbDepends) -> list[User]:
    statement = select(User)
    users = list(session.exec(statement).all())
    return users


def read_user_id(session: DbDepends, user_id: int) -> User | None:
    statement = select(User).where(User.id == user_id)
    user = session.exec(statement).one_or_none()
    return user


def read_user_login(session: DbDepends, login: str) -> User | None:
    statement = select(User).where(User.login == login)
    user = session.exec(statement).one_or_none()
    return user
