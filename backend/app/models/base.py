from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


# Definicje Enumów (muszą pasować do tych w bazie)
class LightLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class HumidityLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


# 1. TABELA UŻYTKOWNIKÓW
class User(SQLModel, table=True):
    __tablename__: str = "users"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True, max_length=255)
    password_hash: str = Field(max_length=255)
    username: str = Field(max_length=100)

    # Statystyki
    xp: int = Field(default=0)
    day_streak: int = Field(default=0)
    last_login_at: datetime | None = Field(default=None)

    # Lokalizacja i ustawienia
    location_city: str | None = Field(default=None, max_length=100)

    # JSONB - używamy sa_column dla specyficznych typów Postgresa
    preferences: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "dark_mode": False,
            "care_reminders": True,
            "weather_tips": True,
        },
        sa_column=Column(JSONB),
    )

    # Metadane
    created_at: datetime | None = Field(default_factory=datetime.now)
    deleted_at: datetime | None = Field(default=None)


# 2. KATALOG ROŚLIN
class PlantsCatalog(SQLModel, table=True):
    __tablename__: str = "plants_catalog"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    common_name: str = Field(max_length=150)
    scientific_name: str | None = Field(default=None, max_length=150)

    preferred_sunlight: LightLevel
    preferred_temp_min: int | None = None
    preferred_temp_max: int | None = None
    air_humidity_req: HumidityLevel | None = None
    soil_humidity_req: HumidityLevel | None = None

    prefered_watering_interval_days: int | None = None


# 3. ROŚLINY UŻYTKOWNIKA
class UserPlants(SQLModel, table=True):
    __tablename__: str = "user_plants"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    plant_catalog_id: int | None = Field(default=None, foreign_key="plants_catalog.id")

    custom_name: str | None = Field(default=None, max_length=100)
    note: str | None = None
    photo_url: str | None = None

    created_at: datetime | None = Field(default_factory=datetime.now)
    age: datetime | None = None


# 4. HISTORIA PODLEWANIA
class WateringData(SQLModel, table=True):
    __tablename__: str = "watering_data"  # type: ignore

    # SQLModel wymaga PK, chociaż w SQL jej nie zdefiniowaliśmy jawnie
    # (zazwyczaj dodaje się ID).
    # Tutaj zrobimy obejście, ale lepiej dodać ID w SQL.
    # Zakładając że plant_id + timestamp to klucz:
    plant_id: int = Field(foreign_key="user_plants.id", primary_key=True)
    timestamp_of_watering: datetime = Field(
        default_factory=datetime.now,
        primary_key=True,
    )


# 5. POZIOMY
class LevelsXpRanges(SQLModel, table=True):
    __tablename__: str = "levels_xp_ranges"  # type: ignore
    level_val: int = Field(primary_key=True)
    req_xp: int


# TOKEN (bez zmian lub drobne poprawki)
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
