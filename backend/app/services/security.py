from datetime import UTC, datetime, timedelta
from typing import Annotated

import argon2
import jwt
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.db import SessionDep
from app.models.base import Token, User
from app.services.users import UserServiceDep
from app.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
SECRET_KEY = settings.jwt_secret
ALGORITHM = "HS256"

credentials_exception = HTTPException(
    status_code=401,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

account_deleted = HTTPException(status_code=403, detail="Account has been deleted")


class AuthService:
    def __init__(self, session: SessionDep, users: UserServiceDep) -> None:
        self.s = session
        self.users = users

    def authenticate(self, token) -> User:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            login = payload.get("sub")
            if login is None:
                raise credentials_exception
        except jwt.InvalidTokenError:
            raise credentials_exception from None
        user = self.users.read_user_email(login)
        if user is None:
            raise credentials_exception
        if user.deleted_at is not None:
            raise account_deleted
        return user

    def password_login(self, form_data: OAuth2PasswordRequestForm) -> Token:
        hasher = PasswordHasher()
        user = self.users.read_user_email(form_data.username)
        if user is None:
            raise credentials_exception

        try:
            hasher.verify(user.password_hash, form_data.password)
        except argon2.exceptions.VerifyMismatchError:
            raise credentials_exception from None
        except argon2.exceptions.InvalidHashError:
            raise credentials_exception from None

        if user.deleted_at is not None:
            raise account_deleted

        expire = datetime.now(UTC) + timedelta(weeks=4)
        data = {"sub": user.email, "exp": expire}
        access_token = jwt.encode(data, SECRET_KEY, ALGORITHM)
        return Token(access_token=access_token, refresh_token="", token_type="bearer")


AuthServiceDep = Annotated[AuthService, Depends()]


def get_logged_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    service: AuthServiceDep,
) -> User:
    return service.authenticate(token)


LoggedUserDep = Annotated[User, Depends(get_logged_user)]
