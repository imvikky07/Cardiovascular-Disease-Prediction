from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import prediction, health
from app.core.config import settings

app = FastAPI(
    title="CVD Risk Detection API",
    description="Cardiovascular Disease Risk Prediction using Logistic Regression",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "CVD Risk Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
