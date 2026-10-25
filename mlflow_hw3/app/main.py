import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from app.features import FEATURE_COLUMNS, MIN_HISTORY_ROWS, build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "BrentDirectionModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")


class CandleInput(BaseModel):
    date: datetime | None = None
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_candle(self):
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, close and high")
        return self


class ProcessRequest(BaseModel):
    candles: list[CandleInput] = Field(min_length=MIN_HISTORY_ROWS, max_length=200)

    @model_validator(mode="after")
    def validate_history(self):
        dates = [candle.date for candle in self.candles]
        has_dates = [value is not None for value in dates]
        if any(has_dates) and not all(has_dates):
            raise ValueError("Provide dates for every candle or omit dates for every candle")
        if all(has_dates) and len(set(dates)) != len(dates):
            raise ValueError("Candle dates must be unique")
        return self


class ProcessResponse(BaseModel):
    prediction: int
    direction: str
    probability_up: float | None
    model_name: str
    model_version: str
    model_alias: str
    run_id: str
    history_rows: int


def predict(model: Any, features: pd.DataFrame) -> tuple[int, float | None]:
    prediction = int(np.asarray(model.predict(features)).reshape(-1)[0])
    probability_up = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))[0]
        classes = np.asarray(model.classes_)
        matches = np.flatnonzero(classes == 1)
        if len(matches):
            probability_up = float(probabilities[matches[0]])
    return prediction, probability_up


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        registered_version = await run_in_threadpool(
            client.get_model_version_by_alias, MODEL_NAME, MODEL_ALIAS
        )
        version = str(registered_version.version)
        model_uri = f"models:/{MODEL_NAME}/{version}"
        model = await run_in_threadpool(mlflow.sklearn.load_model, model_uri)
    except Exception as exc:
        logger.exception("model_load_failed model=%s alias=%s", MODEL_NAME, MODEL_ALIAS)
        raise RuntimeError(
            f"Could not load model {MODEL_NAME} with alias {MODEL_ALIAS}. Run training first."
        ) from exc

    app.state.model = model
    app.state.model_name = MODEL_NAME
    app.state.model_alias = MODEL_ALIAS
    app.state.model_version = version
    app.state.run_id = registered_version.run_id
    logger.info(
        "model_loaded model=%s version=%s alias=%s run_id=%s",
        MODEL_NAME,
        version,
        MODEL_ALIAS,
        registered_version.run_id,
    )
    yield


app = FastAPI(title="Brent Direction Inference", lifespan=lifespan)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    return {"status": "ok", "model_version": request.app.state.model_version}


@app.get("/model")
async def model_info(request: Request) -> dict[str, str]:
    return {
        "name": request.app.state.model_name,
        "version": request.app.state.model_version,
        "alias": request.app.state.model_alias,
        "run_id": request.app.state.run_id,
    }


@app.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    rows = [candle.model_dump(exclude_none=True) for candle in payload.candles]
    history = pd.DataFrame(rows)
    features = build_features(history)
    latest_features = features.iloc[[-1]][FEATURE_COLUMNS]
    if latest_features.isna().to_numpy().any():
        raise HTTPException(
            status_code=422,
            detail=f"At least {MIN_HISTORY_ROWS} valid sequential candles are required",
        )

    try:
        prediction, probability_up = await run_in_threadpool(
            predict, request.app.state.model, latest_features
        )
    except Exception as exc:
        logger.exception("inference_failed model_version=%s", request.app.state.model_version)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return ProcessResponse(
        prediction=prediction,
        direction="up" if prediction == 1 else "down_or_unchanged",
        probability_up=probability_up,
        model_name=request.app.state.model_name,
        model_version=request.app.state.model_version,
        model_alias=request.app.state.model_alias,
        run_id=request.app.state.run_id,
        history_rows=len(payload.candles),
    )
