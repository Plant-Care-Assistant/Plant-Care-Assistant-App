from pydantic import Field
from pydantic_settings import BaseSettings

DEFAULT_IMAGE = "1,01e49b6671"


class Settings(BaseSettings):
    db_url: str = Field(validation_alias="DATABASE_URL")
    blob_url: str = Field(validation_alias="BLOBSTORAGE_URL")
    jwt_secret: str = Field(validation_alias="JWT_SECRET")


settings = Settings()  # type: ignore[reportMissingParameterType]
