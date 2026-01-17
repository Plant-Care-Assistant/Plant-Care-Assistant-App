from datetime import UTC, datetime
from typing import Annotated

from argon2 import PasswordHasher
from fastapi import Depends, HTTPException
from sqlmodel import select

from app.db import SessionDep
from app.models.base import User
from app.models.requests import (
    UserCreate,
    UserPreferences,
    UserUpdate,
)


class UserService:
    def __init__(self, session: SessionDep) -> None:
        self.s = session

    def read_users(self) -> list[User]:
        statement = select(User)
        return list(self.s.exec(statement).all())

    def read_user_email(self, email: str) -> User | None:
        statement = select(User).where(User.email == email)
        return self.s.exec(statement).one_or_none()

    def read_user_id(self, user_id: int) -> User | None:
        return self.s.get(User, user_id)

    def create_user(self, body: UserCreate) -> User:
        if self.read_user_email(body.email) is not None:
            raise HTTPException(409, "Email already in use")

        ph = PasswordHasher()
        password_hash = ph.hash(body.password)
        username = body.username or body.email.split("@")[0]

        new_user = User(
            email=body.email,
            password_hash=password_hash,
            username=username,
        )

        self.s.add(new_user)
        self.s.commit()
        self.s.refresh(new_user)
        return new_user

    def update_user(self, user: User, new_values: UserUpdate) -> User:
        new_user = user.model_copy(update=new_values.model_dump(exclude_none=True))
        self.s.add(new_user)
        self.s.commit()
        return new_user

    def delete_user(self, user) -> None:
        user.deleted_at = datetime.now(UTC)
        self.s.add(user)
        self.s.commit()

    def update_preferences(
        self, user: User, new_prefs: UserPreferences,
    ) -> UserPreferences:
        user.preferences = new_prefs.model_dump()
        self.s.add(user)
        self.s.commit()
        self.s.refresh(user)
        return UserPreferences.model_validate(user.preferences)


UserServiceDep = Annotated[UserService, Depends()]
