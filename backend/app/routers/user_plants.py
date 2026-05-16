from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.models.requests import (
    CareEventCreate,
    CareHistoryPublic,
    UserPlantCreate,
    UserPlantImagePublic,
    UserPlantPublic,
    UserPlantUpdate,
)
from app.services.security import LoggedUserDep
from app.services.user_plants import UserPlantServiceDep

router = APIRouter(prefix="/my-plants", tags=["user_plants"])

UPSD = UserPlantServiceDep


@router.get("", response_model=list[UserPlantPublic])
@router.get("/", response_model=list[UserPlantPublic])
def read_user_plants(user: LoggedUserDep, service: UserPlantServiceDep):
    return service.read_plants(user)


@router.post("", response_model=UserPlantPublic)
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
    return Response(status_code=204)


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
def read_user_plant_tasks(plant_id: int):  # noqa: ARG001
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/{plant_id}/tasks")
def create_user_plant_tasks(plant_id: int):  # noqa: ARG001
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{plant_id}/images", response_model=list[UserPlantImagePublic])
def list_user_plant_images(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    return service.list_images(user, plant_id)


@router.post("/{plant_id}/images", response_model=UserPlantImagePublic)
def upload_user_plant_image(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
    file: Annotated[UploadFile, File()],
):
    return service.add_image(user, plant_id, file)


@router.get("/{plant_id}/images/{image_id}")
def read_user_plant_image_blob(
    plant_id: int,
    image_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    fid = service.get_image_fid(user, plant_id, image_id)
    vid, bid = fid.split(",", 1)
    response = Response()
    response.headers["X-Accel-Redirect"] = f"/blob/{vid}/{bid}/plant.jpg"
    response.headers["Content-Disposition"] = "inline; filename=plant.jpg"
    return response


@router.delete("/{plant_id}/images/{image_id}")
def delete_user_plant_image_record(
    plant_id: int,
    image_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    service.delete_image_record(user, plant_id, image_id)
    return Response(status_code=204)


@router.post("/{plant_id}/water")
def record_user_plant_watering(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    """Legacy alias kept for back-compat; equivalent to POST /care {type:water}."""
    service.record_watering(user, plant_id)
    return Response(status_code=204)


@router.post("/{plant_id}/care")
def record_user_plant_care(
    plant_id: int,
    body: CareEventCreate,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
):
    service.record_care(user, plant_id, body.type)
    return Response(status_code=204)


@router.get("/{plant_id}/care-history", response_model=CareHistoryPublic)
def read_user_plant_care_history(
    plant_id: int,
    user: LoggedUserDep,
    service: UserPlantServiceDep,
    days: int = 30,
):
    return service.care_history(user, plant_id, days=days)
