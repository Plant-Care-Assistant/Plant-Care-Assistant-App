from uuid import UUID, uuid4
from datetime import datetime
from sqlmodel import Field, SQLModel
from pydantic import BaseModel

class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    login: str = Field(index=True)
    password: str
    email: str
    created_at: datetime
    __tablename__: str = "users"  #  type: ignore

class Plant(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    common_name: str
    species_name: str
    image_reference: str
    description: str
    __tablename__: str = "plant_info"  #  type: ignore

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
