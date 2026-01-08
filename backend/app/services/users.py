from argon2 import PasswordHasher
from sqlmodel import select

from app.db import DbDepends
from app.models.base import User
from app.models.requests import UserCreate


def read_users(session: DbDepends) -> list[User]:
    statement = select(User)
    return list(session.exec(statement).all())


def read_user_id(session: DbDepends, user_id: int) -> User | None:
    statement = select(User).where(User.id == user_id)
    return session.exec(statement).one_or_none()


# Zmiana nazwy funkcji i pola wyszukiwania
def read_user_by_email(session: DbDepends, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    return session.exec(statement).one_or_none()


def create_user(session: DbDepends, user_create: UserCreate) -> User:
    ph = PasswordHasher()

    # Generowanie username z emaila (np. jan@op.pl -> jan)
    # Jeśli username nie został podany w requeście (jest opcjonalny w UserCreate)
    # Uwaga: Musisz dostosować UserCreate w requests.py,
    # jeśli username ma być opcjonalny
    base_username = user_create.email.split("@")[0]

    # Hashowanie hasła
    hashed_password = ph.hash(user_create.password)

    new_user = User(
        email=user_create.email,
        password_hash=hashed_password,  # Zapisujemy hash, nie czyste hasło!
        username=base_username,
        xp=0,
        day_streak=0,
    )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user
