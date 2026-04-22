from app.models.base import GameAction
from app.models.requests import (
    GamificationActionBody,
    UserActionResponse,
    UserGamificationReport,
)
from app.services.gamification import GamificationServiceDep
from app.services.security import LoggedUserDep
from fastapi import APIRouter, Query, Response

router = APIRouter(prefix="/gamification", tags=["gamification"])


@router.get("/me", response_model=UserGamificationReport)
def read_stats(user: LoggedUserDep, service: GamificationServiceDep):
    return service.user_report(user)


@router.post("/events", response_model=UserActionResponse)
def create_event(
    body: GamificationActionBody, user: LoggedUserDep, service: GamificationServiceDep
):
    return service.handle_action(
        user, GameAction(body.action_id), body.client_tz_offset_min
    )


@router.get("/actions")
def read_actions(service: GamificationServiceDep):
    pass
