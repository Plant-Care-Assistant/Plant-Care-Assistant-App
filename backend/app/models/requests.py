from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.base import CareType, HumidityLevel, LightLevel


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
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
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

    scientific_name: str | None = None
    preferred_sunlight: LightLevel | None = None
    preferred_temp_min: int | None = None
    preferred_temp_max: int | None = None
    air_humidity_req: HumidityLevel | None = None
    soil_humidity_req: HumidityLevel | None = None
    preferred_watering_interval_days: int | None = None

    last_health_label: str | None = None
    last_health_confidence: float | None = None
    last_health_check_at: datetime | None = None
    last_diseases: list[dict[str, Any]] | None = None


class UserPlantUpdate(BaseModel):
    plant_catalog_id: int | None = None
    custom_name: str | None = None
    note: str | None = None
    sprouted_at: datetime | None = None

    scientific_name: str | None = None
    preferred_sunlight: LightLevel | None = None
    preferred_temp_min: int | None = None
    preferred_temp_max: int | None = None
    air_humidity_req: HumidityLevel | None = None
    soil_humidity_req: HumidityLevel | None = None
    preferred_watering_interval_days: int | None = None

    last_health_label: str | None = None
    last_health_confidence: float | None = None
    last_health_check_at: datetime | None = None
    last_diseases: list[dict[str, Any]] | None = None


class UserPlantPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    plant_catalog_id: int | None = None

    custom_name: str | None = None
    note: str | None = None

    created_at: datetime
    sprouted_at: datetime | None = None

    scientific_name: str | None = None
    preferred_sunlight: LightLevel | None = None
    preferred_temp_min: int | None = None
    preferred_temp_max: int | None = None
    air_humidity_req: HumidityLevel | None = None
    soil_humidity_req: HumidityLevel | None = None
    preferred_watering_interval_days: int | None = None

    last_health_label: str | None = None
    last_health_confidence: float | None = None
    last_health_check_at: datetime | None = None
    last_diseases: list[dict[str, Any]] | None = None

    # Care-urgency fields populated by the service from watering_data. None
    # when the plant has never been watered yet.
    last_watered_at: datetime | None = None
    # Days remaining until the next watering is due (0 = due today/overdue).
    # None when there's no watering interval set and no history.
    days_until_water: int | None = None


class UserPlantImagePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_plant_id: int
    fid: str
    uploaded_at: datetime


class CareEventPublic(BaseModel):
    timestamp: datetime
    type: CareType


class DailyCarePublic(BaseModel):
    """One day in the Weekly Care 7-day strip. `types` is the set of care
    activities the user logged that day (empty list = no care recorded)."""

    date: str  # ISO YYYY-MM-DD
    types: list[CareType]


class CareHistoryPublic(BaseModel):
    """Care history snapshot for plant detail screen widgets."""

    waterings: list[datetime]
    events: list[CareEventPublic]
    current_streak_days: int
    unique_days_last_week: int
    daily_last_week: list[DailyCarePublic]


class CareEventCreate(BaseModel):
    type: CareType


class PlantPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    common_name: str
    scientific_name: str | None
    plantsnet_id: str | None
    preferred_sunlight: LightLevel
    preferred_temp_min: int | None
    preferred_temp_max: int | None
    air_humidity_req: HumidityLevel | None
    soil_humidity_req: HumidityLevel | None

    preferred_watering_interval_days: int | None = None


class UserGamificationReport(BaseModel):
    xp: int
    level: int
    counters: dict[str, int]
    flags: dict[str, bool]
    unlocked_achievement_ids: list[str]
    current_streak: int
    longest_streak: int
    last_active_date: datetime | None


class GamificationActionBody(BaseModel):
    action_id: str
    client_tz_offset_min: int


class UserActionResponse(BaseModel):
    snapshot: UserGamificationReport
    xp_awarded: int
    newly_unlocked: list[str]
