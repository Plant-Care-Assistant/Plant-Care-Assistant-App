from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_IMAGE = "1,01e49b6671"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    db_url: str = Field(validation_alias="DATABASE_URL")
    blob_url: str = Field(validation_alias="BLOBSTORAGE_URL")
    blob_backend_url: str = Field(
        default="100.100.1.1:8333",
        validation_alias="BLOBSTORAGE_BACKEND_URL",
    )
    jwt_secret: str = Field(validation_alias="JWT_SECRET")


settings = Settings()  # type: ignore[reportMissingParameterType]
