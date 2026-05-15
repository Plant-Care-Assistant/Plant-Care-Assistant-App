import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import httpx
from fastapi import Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from app.db import SessionDep
from app.models.base import CareType, User, UserPlant, UserPlantImage, CareEvent
from app.models.requests import UserPlantCreate, UserPlantUpdate
from app.settings import DEFAULT_IMAGE, settings

# Fallback watering interval when neither the plant override nor the catalog
# specifies one. Matches the frontend default so values agree.
DEFAULT_WATERING_INTERVAL_DAYS = 7

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

    def _last_waterings(self, plant_ids: list[int]) -> dict[int, datetime]:
        """Single GROUP BY query → {plant_id: latest watering timestamp}.

        Pulls only `care_type='water'` rows; misting/fertilizing don't reset
        the watering schedule. Plants without any watering history are absent
        from the map.
        """
        if not plant_ids:
            return {}
        rows = self.s.exec(
            select(
                CareEvent.plant_id,
                func.max(CareEvent.timestamp_of_watering),
            )
            .where(CareEvent.plant_id.in_(plant_ids))
            .where(CareEvent.care_type == CareType.water)
            .group_by(CareEvent.plant_id),
        ).all()
        return {pid: ts for pid, ts in rows}

    @staticmethod
    def _enrich(plant: UserPlant, last_watered_at: datetime | None) -> dict[str, Any]:
        """Convert a UserPlant ORM row to a dict + care-urgency fields.

        Returning a dict (not the ORM object) so the Pydantic response model
        picks up the extra computed fields without requiring them on the
        SQLModel class.
        """
        data: dict[str, Any] = plant.model_dump()
        data["last_watered_at"] = last_watered_at

        interval = plant.preferred_watering_interval_days or DEFAULT_WATERING_INTERVAL_DAYS
        if last_watered_at is None:
            # Never watered — treat the plant as due in `interval` days from
            # now (i.e. give the user the full interval to do the first watering).
            data["days_until_water"] = interval
        else:
            days_since = (datetime.now(UTC) - last_watered_at).days
            data["days_until_water"] = max(0, interval - days_since)
        return data

    def read_plant(self, user: User, plant_id: int) -> dict[str, Any]:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")
        last = self._last_waterings([plant_id]).get(plant_id)
        return self._enrich(plant, last)

    def read_plants(self, user: User) -> list[dict[str, Any]]:
        self._check_user(user)
        plants = list(
            self.s.exec(select(UserPlant).where(UserPlant.user_id == user.id)).all(),
        )
        last_map = self._last_waterings([p.id for p in plants if p.id is not None])
        return [self._enrich(p, last_map.get(p.id)) for p in plants if p.id is not None]

    def create_plant(self, user: User, body: UserPlantCreate) -> dict[str, Any]:
        user_id = self._check_user(user)
        new_plant = UserPlant(user_id=user_id, **body.model_dump(exclude_none=True))
        self.s.add(new_plant)
        self.s.commit()
        self.s.refresh(new_plant)
        # Brand-new plant has no waterings yet.
        return self._enrich(new_plant, None)

    def update_plant(
        self,
        user: User,
        plant_id: int,
        body: UserPlantUpdate,
    ) -> dict[str, Any]:
        self._check_user(user)
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        new_plant = plant.model_copy(update=body.model_dump(exclude_none=True))
        self.s.add(new_plant)
        self.s.commit()
        self.s.refresh(new_plant)
        last = self._last_waterings([plant_id]).get(plant_id)
        return self._enrich(new_plant, last)

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

    # === Care history (watering + other care activities) ===

    def record_watering(self, user: User, plant_id: int) -> CareEvent:
        """Back-compat wrapper used by the legacy /water endpoint."""
        return self.record_care(user, plant_id, CareType.water)

    def record_care(
        self,
        user: User,
        plant_id: int,
        care_type: CareType,
    ) -> CareEvent:
        self._ensure_owned(user, plant_id)
        row = CareEvent(
            plant_id=plant_id,
            timestamp_of_watering=datetime.now(UTC),
            care_type=care_type,
        )
        self.s.add(row)
        self.s.commit()
        self.s.refresh(row)
        return row

    def care_history(self, user: User, plant_id: int, days: int = 30) -> dict:
        """Return care events within the last `days` plus computed widgets.

        - waterings: timestamps of watering events only (back-compat for the
          "last watered" card on the detail screen).
        - events: every care event in the window with its type.
        - current_streak_days: consecutive trailing days with ≥ 1 care event of
          any type (water/mist/fertilize/...).
        - unique_days_last_week: distinct days with ≥ 1 care event in last 7.
        - daily_last_week: ordered list of 7 entries (oldest → today) each
          containing date + set of care types performed that day, for the
          Weekly Care grid.
        """
        self._ensure_owned(user, plant_id)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        st = (
            select(CareEvent)
            .where(CareEvent.plant_id == plant_id)
            .where(CareEvent.timestamp_of_watering >= cutoff)
            .order_by(CareEvent.timestamp_of_watering.desc())
        )
        rows = list(self.s.exec(st).all())

        events = [
            {"timestamp": r.timestamp_of_watering, "type": r.care_type.value}
            for r in rows
        ]
        waterings = [
            r.timestamp_of_watering for r in rows if r.care_type == CareType.water
        ]

        # Day-bucket every care event (any type) for streak + weekly metrics.
        cared_days_by_type: dict = {}
        for r in rows:
            d = r.timestamp_of_watering.date()
            cared_days_by_type.setdefault(d, set()).add(r.care_type.value)
        cared_days = set(cared_days_by_type.keys())

        today = datetime.now(UTC).date()
        streak = 0
        cursor = today
        while cursor in cared_days:
            streak += 1
            cursor -= timedelta(days=1)

        week_cutoff = today - timedelta(days=6)
        unique_days_last_week = len({d for d in cared_days if d >= week_cutoff})

        # Build a fixed 7-day strip ending today so the frontend doesn't have
        # to fill gaps; missing days have an empty `types` list.
        daily_last_week = [
            {
                "date": (week_cutoff + timedelta(days=i)).isoformat(),
                "types": sorted(
                    cared_days_by_type.get(week_cutoff + timedelta(days=i), set())
                ),
            }
            for i in range(7)
        ]

        return {
            "waterings": waterings,
            "events": events,
            "current_streak_days": streak,
            "unique_days_last_week": unique_days_last_week,
            "daily_last_week": daily_last_week,
        }


UserPlantServiceDep = Annotated[UserPlantService, Depends()]
