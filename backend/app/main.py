from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, misc, plants, user_plants, users

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(plants.router)
app.include_router(user_plants.router)
app.include_router(misc.router)
