from fastapi import APIRouter

router = APIRouter(tags=["others"])


@router.get("/recognize")
def recognize_image():
    pass
