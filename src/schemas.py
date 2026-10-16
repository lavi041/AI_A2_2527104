"""Typed request/response schemas shared by the CLI and the FastAPI service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FarmRecord(BaseModel):
    """One farm record. Strict types, ranges = ranges of the training data."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rainfall_mm: float = Field(ge=600, le=1400, description="Seasonal rainfall (mm)")
    avg_temp_c: float = Field(ge=12, le=26, description="Average temperature (deg C)")
    soil_moisture_pct: float = Field(ge=20, le=80, description="Soil moisture (%)")
    fertilizer_kg: float = Field(ge=0, le=250, description="Fertilizer applied (kg)")
    field_size_ha: float = Field(ge=0.2, le=4.0, description="Field size (hectares)")
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


class PredictionResponse(BaseModel):
    yield_estimate_tons: float
    high_risk_prediction: int
    high_risk_label: str
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    verification_code: str
    disclaimer: str


def format_errors(errors: list[dict]) -> list[dict]:
    """Compact, safe error list (never echoes the submitted values)."""
    return [
        {"field": ".".join(str(p) for p in e.get("loc", []) if p != "body") or "body",
         "message": e.get("msg", "invalid"), "type": e.get("type", "")}
        for e in errors
    ]
