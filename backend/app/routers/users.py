from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.base import User
from app.models.requests import UserPublic
from app.services.security import LoggedUserDepends
from app.services.users import read_users

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=list[UserPublic])
def read(users: Annotated[list[User], Depends(read_users)]) -> list[User]:
    return users


@router.get("/me", response_model=UserPublic)
def read_current_user(user: LoggedUserDepends) -> User:
    return user
