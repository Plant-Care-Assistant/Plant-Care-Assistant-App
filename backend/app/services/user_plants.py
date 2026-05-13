import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import select

from app.db import SessionDep
from app.models.base import User, UserPlant, UserPlantImage, WateringData
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

    # === Gallery: multiple images per plant ===

    def _ensure_owned(self, user: User, plant_id: int) -> UserPlant:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")
        return plant

    def _upload_blob(self, file: UploadFile) -> str:
        """Upload a file to SeaweedFS and return its fid. Raises HTTPException on failure."""
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

        return slot.fid

    def list_images(self, user: User, plant_id: int) -> list[UserPlantImage]:
        self._ensure_owned(user, plant_id)
        st = (
            select(UserPlantImage)
            .where(UserPlantImage.user_plant_id == plant_id)
            .order_by(UserPlantImage.uploaded_at.desc())
        )
        return list(self.s.exec(st).all())

    def add_image(self, user: User, plant_id: int, file: UploadFile) -> UserPlantImage:
        self._ensure_owned(user, plant_id)
        fid = self._upload_blob(file)
        row = UserPlantImage(user_plant_id=plant_id, fid=fid)
        self.s.add(row)
        self.s.commit()
        self.s.refresh(row)
        return row

    def delete_image_record(self, user: User, plant_id: int, image_id: int) -> None:
        self._ensure_owned(user, plant_id)
        row = self.s.get(UserPlantImage, image_id)
        if row is None or row.user_plant_id != plant_id:
            raise HTTPException(404, "Image not found")
        try:
            httpx.delete(f"http://{BLOB_BACKEND_URL}/{row.fid}", timeout=HTTP_TIMEOUT)
        except httpx.HTTPError as e:
            logger.warning("Failed to delete blob %s: %s", row.fid, e)
        self.s.delete(row)
        self.s.commit()

    def get_image_fid(self, user: User, plant_id: int, image_id: int) -> str:
        self._ensure_owned(user, plant_id)
        row = self.s.get(UserPlantImage, image_id)
        if row is None or row.user_plant_id != plant_id:
            raise HTTPException(404, "Image not found")
        return row.fid

    # === Care history (watering) ===

    def record_watering(self, user: User, plant_id: int) -> WateringData:
        self._ensure_owned(user, plant_id)
        row = WateringData(plant_id=plant_id, timestamp_of_watering=datetime.now(UTC))
        self.s.add(row)
        self.s.commit()
        self.s.refresh(row)
        return row

    def care_history(self, user: User, plant_id: int, days: int = 30) -> dict:
        """Return waterings within the last `days` plus computed widgets:
        - current_streak_days: count of consecutive trailing days with ≥1 watering
        - unique_days_last_week: distinct days with ≥1 watering in the last 7 days
        """
        self._ensure_owned(user, plant_id)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        st = (
            select(WateringData)
            .where(WateringData.plant_id == plant_id)
            .where(WateringData.timestamp_of_watering >= cutoff)
            .order_by(WateringData.timestamp_of_watering.desc())
        )
        rows = list(self.s.exec(st).all())
        waterings = [r.timestamp_of_watering for r in rows]

        # Day-bucket the timestamps in UTC.
        watered_days = {ts.date() for ts in waterings}

        today = datetime.now(UTC).date()
        streak = 0
        cursor = today
        while cursor in watered_days:
            streak += 1
            cursor -= timedelta(days=1)

        week_cutoff = today - timedelta(days=6)
        unique_days_last_week = len({d for d in watered_days if d >= week_cutoff})

        return {
            "waterings": waterings,
            "current_streak_days": streak,
            "unique_days_last_week": unique_days_last_week,
        }


UserPlantServiceDep = Annotated[UserPlantService, Depends()]
