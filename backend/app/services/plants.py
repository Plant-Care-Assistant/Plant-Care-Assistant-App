from typing import Annotated

from fastapi import Depends, HTTPException
from rapidfuzz import process
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
        plants = list(self.s.exec(select(Plant)).all())
        mapping = {}
        for plant in plants:
            mapping[plant.common_name.lower()] = plant
            if plant.scientific_name is not None:
                mapping[plant.scientific_name.lower()] = plant

        choices = mapping.keys()
        result = process.extract(name.lower(), choices, limit=5)
        top_plants = []
        for r in result:
            if mapping[r[0]] not in top_plants:
                top_plants.append(mapping[r[0]])
        return top_plants

    def read_plant_image(self, plant_id: int) -> str:
        plant = self.s.get(Plant, plant_id)
        if plant is None:
            raise HTTPException(404, "Plant not found")

        if plant.fid is None:
            return DEFAULT_IMAGE
        return plant.fid


PlantServiceDep = Annotated[PlantService, Depends()]
