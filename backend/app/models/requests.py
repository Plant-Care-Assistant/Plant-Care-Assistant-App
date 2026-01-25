from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field

from app.models.base import HumidityLevel, LightLevel


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


class UserPlantCreate(BaseModel):
    plant_catalog_id: int | None = None
    custom_name: str | None = None
    note: str | None = None
    sprouted_at: datetime | None = None

    scientific_name: str | None
    preferred_sunlight: LightLevel | None = None
    preferred_temp_min: int | None = None
    preferred_temp_max: int | None = None
    air_humidity_req: HumidityLevel | None = None
    soil_humidity_req: HumidityLevel | None = None
    preferred_watering_interval_days: int | None = None


class UserPlantUpdate(BaseModel):
    plant_catalog_id: int | None = None
    custom_name: str | None = None
    note: str | None = None
    sprouted_at: datetime | None = None

    scientific_name: str | None
    preferred_sunlight: LightLevel | None = None
    preferred_temp_min: int | None = None
    preferred_temp_max: int | None = None
    air_humidity_req: HumidityLevel | None = None
    soil_humidity_req: HumidityLevel | None = None
    preferred_watering_interval_days: int | None = None


class UserPlantPublic(BaseModel):
    id: int
    plant_catalog_id: int

    custom_name: str | None
    note: str | None

    created_at: datetime
    sprouted_at: datetime | None

    scientific_name: str | None
    preferred_sunlight: LightLevel | None
    preferred_temp_min: int | None
    preferred_temp_max: int | None
    air_humidity_req: HumidityLevel | None
    soil_humidity_req: HumidityLevel | None
    preferred_watering_interval_days: int | None


class PlantPublic(BaseModel):
    id: int
    common_name: str
    scientific_name: str | None
    preferred_sunlight: LightLevel
    preferred_temp_min: int | None
    preferred_temp_max: int | None
    air_humidity_req: HumidityLevel | None
    soil_humidity_req: HumidityLevel | None

    preferred_watering_interval_days: int | None = None
