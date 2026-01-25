from fastapi import APIRouter

from app.models.requests import (
    AchievementPublic,
    UserPreferences,
    UserPublic,
    UserStats,
    UserUpdate,
    WeeklyChallenge,
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
    return service.delete_user(user)


@router.patch("/me", response_model=UserPublic)
def patch_current_user(body: UserUpdate, user: LoggedUserDep, service: UserServiceDep):
    return service.update_user(user, body)


@router.get("/me/stats", response_model=UserStats)
def read_current_user_stats(user: LoggedUserDep):
    level = max(1, user.xp // 100 + 1)
    xp_to_next = 100 - (user.xp % 100)
    return UserStats(
        name=user.username,
        level=level,
        xp=user.xp,
        xp_to_next_level=xp_to_next,
        streak=user.day_streak,
        health_score=0,
        weekly_challenge=WeeklyChallenge(
            completed=0,
            total=7,
            description="Water 7 plants this week",
        ),
        achievements=[],
    )


@router.get("/me/achievement", response_model=list[AchievementPublic])
def read_current_user_achievements():
    return []


@router.get("/me/settings", response_model=UserPreferences)
def read_current_user_settings(user: LoggedUserDep):
    return user.preferences


@router.put("/me/settings", response_model=UserPreferences)
def update_current_user_settings(
    body: UserPreferences,
    user: LoggedUserDep,
    service: UserServiceDep,
):
    return service.update_preferences(user, body)
