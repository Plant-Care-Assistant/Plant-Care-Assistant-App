from typing import Annotated

from fastapi import APIRouter, Query, Response

from app.models.base import PageParams
from app.models.requests import PlantPublic
from app.services.plants import PlantServiceDep

router = APIRouter(prefix="/plants", tags=["plants"])

PageParamsDep = Annotated[PageParams, Query()]
SearchParamsDep = Annotated[str, Query()]


@router.get("/", response_model=list[PlantPublic])
def read_plants(p: PageParamsDep, service: PlantServiceDep):
    return service.read_plants(p)


@router.get("/search", response_model=list[PlantPublic])
def read_plant_name(q: SearchParamsDep, service: PlantServiceDep):
    return service.read_plant_name(q)


@router.get("/{plant_id}", response_model=PlantPublic)
def read_plant(plant_id: int, service: PlantServiceDep):
    return service.read_plant(plant_id)


@router.get("/{plant_id}/image")
def plant_image(plant_id: int, service: PlantServiceDep):
    # Volumen ID, Blob ID
    vid, bid = service.read_plant_image(plant_id).split(",", 1)

    response = Response()
    redirect = f"/blob/{vid}/{bid}/plant.jpg"
    response.headers["X-Accel-Redirect"] = redirect
    response.headers["Content-Disposition"] = "inline; filename=plant.jpg"
    return response
