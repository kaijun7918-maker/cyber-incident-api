"""FastAPI service for the exported Part I Spark Random Forest model."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyber-incident-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "deployment_random_forest.json.gz"


class PortableSparkRandomForest:
    """Evaluate the exact fitted Spark ML trees without starting Spark/JVM."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise RuntimeError(f"Exported model not found at {model_path}")

        with gzip.open(model_path, "rt", encoding="utf-8") as file:
            artifact: dict[str, Any] = json.load(file)

        if artifact.get("format") != "spark-random-forest-portable-v1":
            raise RuntimeError("Unsupported exported model format.")

        self.source = str(artifact["source"])
        self.spark_version = str(artifact["spark_version"])
        self.categorical_features = artifact["categorical_features"]
        self.trees = artifact["trees"]
        self.tree_weights = artifact["tree_weights"]

        if not self.trees or len(self.trees) != len(self.tree_weights):
            raise RuntimeError("The exported forest is incomplete.")

    def build_features(self, incident: dict[str, Any]) -> list[float]:
        affected_users = float(incident["affected_users"])
        financial_loss = float(incident["financial_loss_million"])

        features = [
            float(int(incident["year"]) - 2015),
            financial_loss,
            affected_users,
            financial_loss * 1_000_000.0 / affected_users,
        ]

        for specification in self.categorical_features:
            labels = specification["labels"]
            vector_size = int(specification["vector_size"])
            value = str(incident[specification["input_col"]])

            try:
                category_index = labels.index(value)
            except ValueError:
                # Reproduces StringIndexerModel(handleInvalid="keep").
                category_index = len(labels)

            encoded = [0.0] * vector_size
            if 0 <= category_index < vector_size:
                encoded[category_index] = 1.0
            features.extend(encoded)

        return features

    @staticmethod
    def predict_tree(tree: list[list[Any]], features: list[float]) -> float:
        node_index = 0

        while True:
            (
                prediction,
                feature_index,
                number_of_categories,
                left_values,
                left_child,
                right_child,
            ) = tree[node_index]

            if feature_index < 0 or left_child < 0 or right_child < 0:
                return float(prediction)

            feature_value = features[int(feature_index)]

            if int(number_of_categories) == -1:
                goes_left = feature_value <= float(left_values[0])
            else:
                goes_left = feature_value in left_values

            node_index = int(left_child if goes_left else right_child)

    def predict(self, incident: dict[str, Any]) -> float:
        features = self.build_features(incident)
        weighted_prediction = 0.0
        total_weight = 0.0

        for tree, weight in zip(self.trees, self.tree_weights, strict=True):
            numeric_weight = float(weight)
            weighted_prediction += (
                self.predict_tree(tree, features) * numeric_weight
            )
            total_weight += numeric_weight

        if total_weight <= 0:
            raise RuntimeError("The exported forest has no usable tree weights.")

        return weighted_prediction / total_weight


model = PortableSparkRandomForest(MODEL_PATH)
logger.info(
    "Loaded %d exported Spark Random Forest trees from %s",
    len(model.trees),
    MODEL_PATH,
)


class CyberIncidentInput(BaseModel):
    """Validated incident characteristics accepted by /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "country": "China",
                "year": 2024,
                "attack_type": "Ransomware",
                "target_industry": "Banking",
                "financial_loss_million": 50.0,
                "affected_users": 500000,
                "attack_source": "Hacker Group",
                "vulnerability_type": "Unpatched Software",
                "defense_mechanism": "Firewall",
            }
        }
    )

    country: str = Field(min_length=1, max_length=100)
    year: int = Field(ge=2015, le=2024)
    attack_type: str = Field(min_length=1, max_length=100)
    target_industry: str = Field(min_length=1, max_length=100)
    financial_loss_million: float = Field(ge=0)
    affected_users: int = Field(gt=0)
    attack_source: str = Field(min_length=1, max_length=100)
    vulnerability_type: str = Field(min_length=1, max_length=100)
    defense_mechanism: str = Field(min_length=1, max_length=100)

    @field_validator(
        "country",
        "attack_type",
        "target_industry",
        "attack_source",
        "vulnerability_type",
        "defense_mechanism",
        mode="before",
    )
    @classmethod
    def strip_text_values(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class PredictionOutput(BaseModel):
    predicted_resolution_time_hours: float
    predicted_resolution_time_days: float
    model_type: str
    spark_version: str


app = FastAPI(
    title="Cyber Incident Resolution-Time API",
    description=(
        "Loads the portable export of the Spark Random Forest PipelineModel "
        "trained in Part I and predicts incident resolution time in hours."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Cyber Incident Resolution-Time API is running.",
        "documentation": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "Exported Spark ML Random Forest PipelineModel",
        "number_of_trees": len(model.trees),
        "spark_version": model.spark_version,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CyberIncidentInput) -> PredictionOutput:
    """Predict incident resolution time from validated characteristics."""
    try:
        predicted_hours = max(0.0, model.predict(data.model_dump()))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

    return PredictionOutput(
        predicted_resolution_time_hours=round(predicted_hours, 2),
        predicted_resolution_time_days=round(predicted_hours / 24, 2),
        model_type="Spark ML Random Forest Regressor (portable export)",
        spark_version=model.spark_version,
    )
