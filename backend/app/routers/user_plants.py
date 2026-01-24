from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile

from app.models.requests import UserPlantCreate, UserPlantPublic, UserPlantUpdate
from app.services.security import LoggedUserDep
from app.services.user_plants import UserPlantServiceDep

router = APIRouter(prefix="/my-plants", tags=["user_plants"])

UPSD = UserPlantServiceDep


@router.get("/", response_model=list[UserPlantPublic])
def read_user_plants(user: LoggedUserDep, service: UserPlantServiceDep):
    return service.read_plants(user)


@router.post("/", response_model=UserPlantPublic)
def create_user_plant(
    body: UserPlantCreate,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    return service.create_plant(user, body)


@router.get("/{plant_id}", response_model=UserPlantPublic)
def read_user_plant(plant_id: int, user: LoggedUserDep, service: UserPlantServiceDep):
    return service.read_plant(user, plant_id)


@router.get("/{plant_id}/image")
async def read_user_plant_image(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    # Volumen ID, Blob ID
    vid, bid = service.download_image(user, plant_id).split(",", 1)

    response = Response()
    redirect = f"/blob/{vid}/{bid}/plant.jpg"
    response.headers["X-Accel-Redirect"] = redirect
    response.headers["Content-Disposition"] = "inline; filename=plant.jpg"
    return response


@router.post("/{plant_id}/image")
def create_user_plant_image(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
    file: Annotated[UploadFile, File()],
):
    service.upload_image(user, plant_id, file)
    return Response(status_code=204)


@router.delete("/{plant_id}/image")
async def remove_user_plant_image(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    service.delete_image(user, plant_id)


@router.patch("/{plant_id}", response_model=UserPlantPublic)
def update_user_plant(
    body: UserPlantUpdate,
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    return service.update_plant(user, plant_id, body)


@router.delete("/{plant_id}")
def delete_user_plant(plant_id: int, user: LoggedUserDep, service: UserPlantServiceDep):
    service.delete_plant(user, plant_id)
    return Response(status_code=204)


@router.get("/{plant_id}/tasks")
def read_user_plant_tasks(plant_id: int):
    pass


@router.post("/{plant_id}/tasks")
def create_user_plant_tasks(plant_id: int):
    pass
