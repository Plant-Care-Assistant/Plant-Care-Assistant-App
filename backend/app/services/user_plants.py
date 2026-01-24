import logging
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import select

from app.db import SessionDep
from app.models.base import User, UserPlant
from app.models.requests import UserPlantCreate, UserPlantUpdate
from app.settings import DEFAULT_IMAGE, settings

logger = logging.getLogger(__name__)

BLOB_URL = settings.blob_url
BLOB_BACKEND_URL = settings.blob_backend_url
HTTP_TIMEOUT = 30.0


class BlobSlot(BaseModel):
    count: int
    fid: str
    url: str
    publicUrl: str


class UserPlantService:
    def __init__(self, session: SessionDep) -> None:
        self.s = session

    def _check_user(self, user: User) -> int:
        if user.id is None:
            raise HTTPException(401, "User not authenticated")
        return user.id

    def read_plant(self, user: User, plant_id: int) -> UserPlant:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")
        return plant

    def read_plants(self, user: User):
        self._check_user(user)
        statement = select(UserPlant).where(UserPlant.user_id == user.id)
        return list(self.s.exec(statement).all())

    def create_plant(self, user: User, body: UserPlantCreate) -> UserPlant:
        user_id = self._check_user(user)
        new_plant = UserPlant(user_id=user_id, **body.model_dump(exclude_none=True))
        self.s.add(new_plant)
        self.s.commit()
        self.s.refresh(new_plant)
        return new_plant

    def update_plant(
        self,
        user: User,
        plant_id: int,
        body: UserPlantUpdate,
    ) -> UserPlant:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        new_plant = plant.model_copy(update=body.model_dump(exclude_none=True))
        self.s.add(new_plant)
        self.s.commit()
        self.s.refresh(new_plant)
        return new_plant

    def delete_plant(self, user: User, plant_id: int):
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        if plant.fid is not None:
            try:
                url = f"http://{BLOB_BACKEND_URL}/{plant.fid}"
                httpx.delete(url, timeout=HTTP_TIMEOUT)
            except httpx.HTTPError as e:
                logger.warning("Failed to delete blob %s: %s", plant.fid, e)

        self.s.delete(plant)
        self.s.commit()

    def upload_image(self, user: User, plant_id: int, file: UploadFile):
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        try:
            response = httpx.get(f"http://{BLOB_URL}/dir/assign", timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            slot = BlobSlot.model_validate_json(response.text)
        except httpx.HTTPError as e:
            logger.exception("Failed to assign blob slot")
            raise HTTPException(503, f"Blob storage unavailable: {e}") from e

        filename = file.filename or "plantus"
        file.file.seek(0)
        files = {filename: file.file}

        try:
            upload_response = httpx.post(
                f"http://{BLOB_BACKEND_URL}/{slot.fid}",
                files=files,
                timeout=HTTP_TIMEOUT,
            )
            upload_response.raise_for_status()
        except httpx.HTTPError as e:
            logger.exception("Failed to upload image to blob storage")
            raise HTTPException(503, f"Failed to upload image: {e}") from e

        plant.fid = slot.fid
        self.s.add(plant)
        self.s.commit()

    def download_image(self, user: User, plant_id: int) -> str:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        if plant.fid is None:
            return DEFAULT_IMAGE
        return plant.fid

    def delete_image(self, user: User, plant_id: int) -> None:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")
        if plant.fid is None:
            return

        try:
            httpx.delete(f"http://{BLOB_BACKEND_URL}/{plant.fid}", timeout=HTTP_TIMEOUT)
        except httpx.HTTPError as e:
            logger.warning("Failed to delete blob %s: %s", plant.fid, e)

        plant.fid = None
        self.s.add(plant)
        self.s.commit()


UserPlantServiceDep = Annotated[UserPlantService, Depends()]
