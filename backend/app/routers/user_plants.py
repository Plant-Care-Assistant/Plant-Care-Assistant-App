from fastapi import APIRouter

router = APIRouter(prefix="/my-plants", tags=["user_plants"])


@router.get("/")
def read_user_plants():
    pass


@router.post("/")
def create_user_plant():
    pass


@router.get("/{plant_id}")
def read_user_plant(plant_id: int):
    pass


@router.patch("/{plant_id}")
def update_user_plant(plant_id: int):
    pass


@router.delete("/{plant_id}")
def delete_user_plant(plant_id: int):
    pass


@router.get("/{plant_id}/tasks")
def read_user_plant_tasks(plant_id: int):
    pass


@router.post("/{plant_id}/tasks")
def create_user_plant_tasks(plant_id: int):
    pass
