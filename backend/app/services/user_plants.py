from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import select

from app.db import SessionDep
from app.models.base import User, UserPlant
from app.models.requests import UserPlantCreate, UserPlantUpdate
from app.settings import DEFAULT_IMAGE, settings

BLOB_URL = settings.blob_url


class BlobSlot(BaseModel):
    count: int
    fid: str
    url: str
    publicUrl: str


class UserPlantService:
    def __init__(self, session: SessionDep) -> None:
        self.s = session

    def read_plant(self, user: User, plant_id: int) -> UserPlant:
        assert user.id is not None
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")
        return plant

    def read_plants(self, user: User):
        assert user.id is not None
        statement = select(UserPlant).where(UserPlant.user_id == user.id)
        return list(self.s.exec(statement).all())

    def create_plant(self, user: User, body: UserPlantCreate) -> UserPlant:
        assert user.id is not None
        new_plant = UserPlant(user_id=user.id, **body.model_dump(exclude_none=True))
        self.s.add(new_plant)
        self.s.commit()
        self.s.refresh(new_plant)
        return new_plant

    def update_plant(
        self, user: User, plant_id: int, body: UserPlantUpdate,
    ) -> UserPlant:
        assert user.id is not None
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        new_plant = plant.model_copy(update=body.model_dump(exclude_none=True))
        self.s.add(new_plant)
        self.s.commit()
        self.s.refresh(new_plant)
        return new_plant

    def delete_plant(self, user: User, plant_id: int):
        assert user.id is not None
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        if plant.fid is not None:
            url = "100.100.1.1:8333"
            httpx.delete(f"http://{url}/{plant.fid}")

        self.s.delete(plant)
        self.s.commit()

    def upload_image(self, user: User, plant_id: int, file: UploadFile):
        assert user.id is not None
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        response = httpx.get(f"http://{BLOB_URL}/dir/assign").text
        slot = BlobSlot.model_validate_json(response)
        slot.url = "100.100.1.1:8333"

        filename = file.filename or "plantus"
        file.file.seek(0)
        files = {filename: file.file}
        httpx.post(f"http://{slot.url}/{slot.fid}", files=files)

        plant.fid = slot.fid
        self.s.add(plant)
        self.s.commit()

    def download_image(self, user: User, plant_id: int) -> str:
        assert user.id is not None
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")

        if plant.fid is None:
            return DEFAULT_IMAGE
        return plant.fid

    def delete_image(self, user: User, plant_id: int) -> None:
        assert user.id is not None
        plant = self.s.get(UserPlant, plant_id)
        if plant is None or plant.user_id != user.id:
            raise HTTPException(404, "Plant not found")
        if plant.fid is None:
            return

        url = "100.100.1.1:8333"
        httpx.delete(f"http://{url}/{plant.fid}")
        plant.fid = None

        self.s.add(plant)
        self.s.commit()


UserPlantServiceDep = Annotated[UserPlantService, Depends()]
