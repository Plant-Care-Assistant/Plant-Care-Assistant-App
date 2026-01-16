from fastapi import APIRouter

from app.models.requests import (
    UserPreferences,
    UserPreferencesUpdate,
    UserPublic,
    UserUpdate,
)
from app.services.security import LoggedUserDep
from app.services.users import UserServiceDep

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=list[UserPublic])
def read_users(service: UserServiceDep):
    return service.read_users()


@router.get("/me", response_model=UserPublic)
def read_current_user(user: LoggedUserDep):
    return user


@router.delete("/me")
def delete_current_user(user: LoggedUserDep, service: UserServiceDep) -> None:
    return service.delete_user(user.id)


@router.patch("/me", response_model=UserPublic)
def patch_current_user(body: UserUpdate, user: LoggedUserDep, service: UserServiceDep):
    service.update_user(user, body)


@router.get("/me/stats", response_model=UserPublic)
def read_current_user_stats(user: LoggedUserDep):
    pass


@router.get("/me/achievement", response_model=UserPublic)
def read_current_user_achievements(user: LoggedUserDep):
    pass


@router.get("/me/settings", response_model=UserPreferences)
def read_current_user_settings(user: LoggedUserDep):
    return user.preferences


@router.put("/me/settings", response_model=UserPreferences)
def update_current_user_settings(
    body: UserPreferencesUpdate, user: LoggedUserDep, service: UserServiceDep,
):
    return service.update_preferences(user, body)
