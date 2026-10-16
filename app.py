"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import DISCLAIMER
from src.inference import ModelsNotFoundError, load_bundle, predict_record
from src.schemas import FarmRecord, PredictionResponse, format_errors

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    version="1.0.0",
    description="Classroom decision-support prototype (synthetic data). " + DISCLAIMER,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Return a compact 422 error that never echoes the submitted values."""
    code = None
    try:
        code = load_bundle()["meta"]["verification_code"]
    except ModelsNotFoundError:
        pass
    return JSONResponse(status_code=422, content={
        "error": "validation_error", "detail": format_errors(exc.errors()), "verification_code": code})


@app.exception_handler(ModelsNotFoundError)
async def models_missing_handler(request: Request, exc: ModelsNotFoundError):
    return JSONResponse(status_code=503, content={"error": "models_not_ready", "detail": str(exc)})


@app.get("/health")
def health():
    """Liveness + model availability."""
    bundle = load_bundle()
    return {"status": "ok", "models_loaded": True, "verification_code": bundle["meta"]["verification_code"]}


@app.get("/model-info")
def model_info():
    """Model, data and identity information (includes the AI-A2-LLLL verification code)."""
    return load_bundle()["meta"]


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord):
    """Predict yield, high-risk status and cluster for one farm."""
    return predict_record(record)
