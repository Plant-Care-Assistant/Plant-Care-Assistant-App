from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, misc, plants, user_plants, users

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(plants.router, prefix="/api")
app.include_router(user_plants.router, prefix="/api")
app.include_router(misc.router, prefix="/api")
