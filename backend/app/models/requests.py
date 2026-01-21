from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr = Field(..., max_length=64, examples=["ogrodnik@example.com"])
    password: str = Field(..., min_length=6, max_length=64, examples=["tajnehaslo"])
    username: str | None = Field(None, max_length=30, examples=["FoxLvr"])


class UserUpdate(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    location_city: str | None = None
    preferences: dict[str, Any] | None = None


class UserPublic(BaseModel):
    id: int
    username: str
    email: EmailStr
    xp: int
    day_streak: int
    location_city: str | None = None
    created_at: datetime | None = None


class UserPreferences(BaseModel):
    dark_mode: bool = False
    care_reminders: bool = True
    weather_tips: bool = True
