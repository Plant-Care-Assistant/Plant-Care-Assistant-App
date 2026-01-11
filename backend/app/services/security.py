from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.db import DbDepends
from app.models.base import Token, User
from app.services.users import read_user_by_email
from app.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
SECRET_KEY = settings.jwt_secret
ALGORITHM = "HS256"


def logged_user(token: Annotated[str, Depends(oauth2_scheme)], s: DbDepends) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")  # Token przechowuje email w 'sub'
        if email is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception from None

    user = read_user_by_email(s, email)
    if user is None:
        raise credentials_exception
    return user


def authenticate(
    s: DbDepends,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
    hasher = PasswordHasher()

    # Formularz wysyła 'username', ale my traktujemy to jako email
    user = read_user_by_email(s, form_data.username)

    if user is None:
        raise credentials_exception

    try:
        if not hasher.verify(user.password_hash, form_data.password):
            pass
    except Exception as err:
        raise credentials_exception from err

    expire = datetime.now(UTC) + timedelta(weeks=4)
    # Zapisujemy email jako 'sub'
    data = {"sub": user.email, "exp": expire}
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

    # Aktualizacja last_login_at i day_streak powinna nastąpić tutaj
    # (lub w oddzielnym serwisie)
    # ale na razie zostawmy sam auth

    return Token(access_token=access_token, refresh_token="", token_type="bearer")


LoggedUserDepends = Annotated[User, Depends(logged_user)]
AuthenticateDepends = Annotated[Token, Depends(authenticate)]
