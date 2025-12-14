from datetime import datetime
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, examples=["littlegamer"])
    password: str = Field(..., min_length=1, examples=["haslo123"])
    email: str = Field(..., min_length=1, examples=["lg@example.com"])
    game_ids: list[int] | None = Field(default=None, min_length=1)


class UserUpdate(BaseModel):
    username: str | None = Field(default=None, min_length=1, examples=["littlegamer"])
    password: str | None = Field(default=None, min_length=1, examples=["haslo123"])
    email: str | None = Field(default=None, min_length=1, examples=["lg@example.com"])
    game_ids: list[int] | None = Field(default=None, min_length=1)


class UserPublic(BaseModel):
    name: str
    login: str
    created_at: datetime
    email: str
