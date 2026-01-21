from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.models.base import Token, User
from app.models.requests import UserCreate, UserPublic
from app.services.security import AuthServiceDep
from app.services.users import UserServiceDep

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic)
def register(body: UserCreate, service: UserServiceDep) -> User:
    return service.create_user(body)


@router.post("/login")
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], service: AuthServiceDep,
) -> Token:
    return service.password_login(form_data)


@router.post("/logout")
def logout(service: AuthServiceDep) -> None:
    pass
