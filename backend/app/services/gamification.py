from datetime import datetime, timedelta, timezone
from typing import Annotated
from math import floor, sqrt

from fastapi import Depends, HTTPException
from sqlmodel import select
from app.db import SessionDep
from app.models.base import Achievement, GameAction, GamificationData, User
from app.models.requests import UserActionResponse, UserGamificationReport

XP_MAPPING = {
    GameAction.scan_identify: 25,
    GameAction.scan_and_add: 100,
    GameAction.add_plant: 75,
    GameAction.water_plant: 15,
    GameAction.complete_care_task: 25,
    GameAction.water_before_9am: 10,
    GameAction.first_login: 50,
    GameAction.complete_profile: 30,
    GameAction.first_home_visit: 10,
    GameAction.first_collection_visit: 10,
    GameAction.first_scan_visit: 10,
    GameAction.first_profile_visit: 10,
    GameAction.first_theme_change: 15,
    GameAction.daily_login_bonus: 20,
    GameAction.achievement_unlock: 50,
}

COUNTERS = [
    "plants_added",
    "plants_scanned",
    "plants_not_added",
    "plants_watered",
    "care_task_completed",
    "species_owned",
    "species_scanned",
    "waters_before_9am",
]


FLAGS = [
    GameAction.first_login,
    GameAction.complete_profile,
    GameAction.first_home_visit,
    GameAction.first_collection_visit,
    GameAction.first_scan_visit,
    GameAction.first_profile_visit,
    GameAction.first_theme_change,
]


class GamificationService:
    def __init__(self, session: SessionDep) -> None:
        self.s = session

    @staticmethod
    def _derive_level(xp: int) -> int:
        # Rightmost solution to quadratic 25*(x-1)(x+4)=xp
        return floor((-75 + sqrt(5625 + 100 * (100 + xp))) / 50)

    @staticmethod
    def _award_achievements(
        current_achievements: list[str], gd: GamificationData
    ) -> list[str]:
        return []

    def user_report(self, user: User) -> UserGamificationReport:
        st = select(GamificationData).where(GamificationData.user_id == user.id)
        gd = self.s.exec(st).one_or_none()
        if gd is None:
            raise HTTPException(500, "Gamification data not found, big oopsie")

        counters = {c: getattr(gd, c, 0) for c in COUNTERS}
        flags = {f.name: f in gd.flags for f in FLAGS}

        st = select(Achievement).where(Achievement.user_id == user.id)
        achievements = [a.achievement for a in self.s.exec(st).all()]
        return UserGamificationReport(
            xp=gd.xp,
            level=self._derive_level(gd.xp),
            counters=counters,
            flags=flags,
            unlocked_achievement_ids=achievements,
            current_streak=gd.current_streak,
            longest_streak=gd.longest_streak,
            last_active_date=gd.last_activity,
        )

    def handle_action(
        self, user: User, action: GameAction, delta_minutes: int = 0
    ) -> UserActionResponse:
        if user.id is None:
            raise HTTPException(401, "User not authenticated")
        st = select(GamificationData).where(GamificationData.user_id == user.id)
        gd = self.s.exec(st).one_or_none()
        if gd is None:
            raise HTTPException(500, "Gamification data not found, big oopsie")

        xp_awarded = XP_MAPPING[action]

        utc_time = datetime.now(timezone.utc)
        client_time = utc_time + timedelta(minutes=delta_minutes)
        match action:
            case GameAction.scan_identify:
                gd.plants_scanned += 1
                gd.plants_not_added += 1
                gd.species_scanned += 1
            case GameAction.scan_and_add:
                gd.plants_scanned += 1
                gd.plants_added += 1
                gd.species_scanned += 1
                gd.species_owned += 1
            case GameAction.add_plant:
                gd.plants_added += 1
                gd.species_owned += 1
            case GameAction.water_plant:
                gd.plants_watered += 1
            case GameAction.complete_care_task:
                gd.care_task_completed += 1
            case GameAction.water_before_9am:
                if client_time.hour < 9:
                    gd.waters_before_9am += 1
                else:
                    xp_awarded = 0

            case GameAction.daily_login_bonus:
                if (
                    gd.last_login_at is None
                    or utc_time.date() != gd.last_login_at.date()
                ):
                    gd.last_login_at = utc_time
                    gd.current_streak += 1
                else:
                    xp_awarded = 0
            case flag if flag in FLAGS:
                if flag.name not in gd.flags:
                    gd.flags.append(flag.name)
                else:
                    xp_awarded = 0

        st = select(Achievement).where(Achievement.user_id == user.id)
        current_achievements = [a.achievement for a in self.s.exec(st).all()]

        newly_unlocked = self._award_achievements(current_achievements, gd)
        xp_awarded += len(newly_unlocked) * 50
        for achievement in newly_unlocked:
            self.s.add(Achievement(user_id=user.id, achievement=achievement))

        self.s.add(gd)
        self.s.commit()
        self.s.refresh(gd)

        return UserActionResponse(
            snapshot=self.user_report(user),
            xp_awarded=xp_awarded,
            newly_unlocked=newly_unlocked,
        )


GamificationServiceDep = Annotated[GamificationService, Depends()]
