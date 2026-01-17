from fastapi import FastAPI

from app.routers import auth, misc, plants, user_plants, users

app = FastAPI()
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(plants.router)
app.include_router(user_plants.router)
app.include_router(misc.router)
