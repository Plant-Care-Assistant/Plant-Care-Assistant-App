from typing import Annotated

from fastapi import APIRouter, Query

from app.models.base import PageParams
from app.services.plants import PlantServiceDep

router = APIRouter(prefix="/plants", tags=["plants"])

PageParamsDep = Annotated[PageParams, Query()]
SearchParamsDep = Annotated[list[str], Query()]


@router.get("/")
def read_plants(p: PageParamsDep, service: PlantServiceDep):
    return service.read_plants(p)


@router.get("/{plant_id}")
def read_plant(plant_id: int, service: PlantServiceDep):
    return service.read_plant(plant_id)


@router.get("/search")
def read_plants_names(q: SearchParamsDep, service: PlantServiceDep):
    return service.read_plant_names(q)
