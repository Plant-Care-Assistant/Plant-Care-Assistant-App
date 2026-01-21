from typing import Annotated

from fastapi import Depends
from sqlmodel import select

from app.db import SessionDep
from app.models.base import PageParams, Plant


class PlantService:
    def __init__(self, session: SessionDep) -> None:
        self.s = session

    def read_plants(self, page: PageParams) -> list[Plant]:
        statement = select(Plant).offset(page.offset).limit(page.limit)
        return list(self.s.exec(statement).all())

    def read_plant(self, plant_id: int) -> Plant | None:
        return self.s.get(Plant, plant_id)

    def read_plant_names(self, names: list[str]) -> list[Plant]:
        pass


PlantServiceDep = Annotated[PlantService, Depends()]
