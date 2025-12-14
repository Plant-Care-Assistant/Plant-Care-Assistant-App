from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_url: PostgresDsn = Field(validation_alias="DATABASE_URL")
    jwt_secret: str = Field(validation_alias="JWT_SECRET")
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore[reportMissingParameterType]
