# ruff: noqa: PLR2004
"""Achievement unlock rules.

Mirrors the frontend catalog in `frontend/src/lib/data/achievements.ts`. Ids
here MUST match the ids used by the frontend so the unlock-toast lookup works
on the client side.
"""

from collections.abc import Callable

from app.models.base import GamificationData

_Predicate = Callable[[GamificationData], bool]

COUNTER_AND_STREAK_ACHIEVEMENTS: list[tuple[str, _Predicate]] = [
    # Collection
    ("first-sprout", lambda gd: gd.plants_added >= 1),
    ("small-garden", lambda gd: gd.plants_added >= 5),
    ("green-house", lambda gd: gd.plants_added >= 10),
    ("jungle-keeper", lambda gd: gd.plants_added >= 20),
    # Scanning
    ("curious-eye", lambda gd: gd.plants_scanned >= 1),
    ("plant-detective", lambda gd: gd.plants_scanned >= 10),
    ("field-researcher", lambda gd: gd.plants_scanned_not_added >= 25),
    ("encyclopedia", lambda gd: gd.species_scanned >= 50),
    # Care & Watering
    ("first-drop", lambda gd: gd.plants_watered >= 1),
    ("hydration-hero", lambda gd: gd.plants_watered >= 25),
    ("perfect-schedule", lambda gd: gd.care_tasks_completed >= 50),
    # Streaks
    ("getting-started", lambda gd: gd.current_streak >= 3),
    ("week-warrior", lambda gd: gd.current_streak >= 7),
    ("monthly-master", lambda gd: gd.current_streak >= 30),
    ("centurion", lambda gd: gd.longest_streak >= 100),
    # Variety
    ("variety-pack", lambda gd: gd.species_owned >= 5),
    ("diverse-garden", lambda gd: gd.species_owned >= 10),
    # Plant Health (proxies — real health tracking is v2)
    ("green-thumb", lambda gd: gd.plants_watered >= 35),
    ("plant-whisperer", lambda gd: gd.care_tasks_completed >= 150),
    # Special — counter-based
    ("early-bird", lambda gd: gd.waters_before_9am >= 1),
]

# (achievement_id, min_level) — evaluated against level derived from xp
LEVEL_ACHIEVEMENTS: list[tuple[str, int]] = [
    ("rookie-gardener", 5),
    ("plant-expert", 10),
    ("botanical-guru", 25),
]

# (achievement_id, GameAction flag-name stored in gd.flags) — flag-based
FLAG_ACHIEVEMENTS: list[tuple[str, str]] = [
    ("welcome-aboard", "first_login"),
    ("profile-pro", "complete_profile"),
]
