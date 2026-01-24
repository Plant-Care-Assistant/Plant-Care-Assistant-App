from typing import Annotated

from fastapi import Depends, HTTPException
from sqlmodel import select

from app.db import SessionDep
from app.models.base import PageParams, Plant
from app.settings import DEFAULT_IMAGE, settings

BLOB_URL = settings.blob_url


class PlantService:
    def __init__(self, session: SessionDep) -> None:
        self.s = session

    def read_plants(self, page: PageParams) -> list[Plant]:
        statement = select(Plant).offset(page.offset).limit(page.limit)
        return list(self.s.exec(statement).all())

    def read_plant(self, plant_id: int) -> Plant:
        plant = self.s.get(Plant, plant_id)
        if plant is None:
            raise HTTPException(404, "Plant not found")
        return plant

    def read_plant_name(self, name: str) -> list[Plant]:
        return []

    def read_plant_image(self, plant_id: int) -> str:
        plant = self.s.get(Plant, plant_id)
        if plant is None:
            raise HTTPException(404, "Plant not found")

        if plant.fid is None:
            return DEFAULT_IMAGE
        return plant.fid


PlantServiceDep = Annotated[PlantService, Depends()]
