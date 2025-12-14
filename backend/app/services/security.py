from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from fastapi import Depends
from fastapi.exceptions import HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.db import DbDepends
from app.models.base import Token, User
from app.services.users import read_user_login
from app.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
SECRET_KEY = settings.jwt_secret


def logged_user(token: Annotated[str, Depends(oauth2_scheme)], s: DbDepends) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        login = payload.get("sub")
        if login is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception from None
    user = read_user_login(s, login)
    if user is None:
        raise credentials_exception
    return user


def authenticate(
    s: DbDepends, form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Incorrent username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
    hasher = PasswordHasher()
    user = read_user_login(s, form_data.username)
    if user is None:
        raise credentials_exception
    if not hasher.verify(user.password, form_data.password):
        raise credentials_exception

    expire = datetime.now(UTC) + timedelta(weeks=4)
    data = {"sub": user.login, "exp": expire}
    access_token = jwt.encode(data, SECRET_KEY, algorithm="HS256")
    return Token(access_token=access_token, refresh_token="", token_type="bearer")


LoggedUserDepends = Annotated[User, Depends(logged_user)]
AuthenticateDepends = Annotated[Token, Depends(authenticate)]
