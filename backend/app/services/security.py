from datetime import datetime, timedelta
from typing import Annotated
from fastapi.exceptions import HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, status
from argon2 import PasswordHasher
import jwt

from app.db import DbDepends
from app.models.base import Token, User
from app.settings import settings
from app.services.users import read_user_login

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
        raise credentials_exception
    user = read_user_login(s, login)
    if user is None:
        raise credentials_exception
    return user


def authenticate(
    s: DbDepends, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
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

    expire = datetime.now() + timedelta(weeks=4)
    data = {"sub": user.login, "exp": expire}
    access_token = jwt.encode(data, SECRET_KEY, algorithm="HS256")
    return Token(access_token=access_token, refresh_token="", token_type="bearer")


LoggedUserDepends = Annotated[User, Depends(logged_user)]
AuthenticateDepends = Annotated[Token, Depends(authenticate)]
