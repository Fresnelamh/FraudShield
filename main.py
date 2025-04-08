from fastapi import FastAPI
from routes.detect import router as detect_router
from routes.alerts import router as alerts_router

app = FastAPI(title="Fraud Detection API", version="1.0")

# Inclusions des routes
app.include_router(detect_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")