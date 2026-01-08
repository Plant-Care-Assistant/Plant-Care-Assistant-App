from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


# Rejestracja wymaga teraz Emaila i Hasła. Username generujemy lub pobieramy.
class UserCreate(BaseModel):
    email: EmailStr = Field(..., examples=["ogrodnik@example.com"])
    password: str = Field(..., min_length=6, examples=["tajnehaslo"])


class UserUpdate(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    location_city: str | None = None
    preferences: dict[str, Any] | None = None


# To co zwracamy frontendowi
class UserPublic(BaseModel):
    id: int
    username: str
    email: EmailStr
    xp: int
    day_streak: int
    location_city: str | None
    created_at: datetime | None
