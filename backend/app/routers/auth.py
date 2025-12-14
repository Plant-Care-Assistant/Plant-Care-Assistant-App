from fastapi import APIRouter

from app.models.base import Token
from app.services.security import AuthenticateDepends

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login_user(token: AuthenticateDepends) -> Token:
    return token
