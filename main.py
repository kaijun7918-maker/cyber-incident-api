"""FastAPI service for the Spark Random Forest resolution-time model."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyber-incident-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "rf"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("PYSPARK_PYTHON", "python3")


def create_spark_session() -> SparkSession:
    """Create one small local Spark session for API inference."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("CyberIncidentResolutionAPI")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config(
            "spark.driver.memory",
            os.getenv("SPARK_DRIVER_MEMORY", "512m"),
        )
        .config("spark.driver.maxResultSize", "64m")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    return session


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Spark model not found at {MODEL_PATH}. "
        "The folder must contain metadata and stages."
    )

spark = create_spark_session()
model = PipelineModel.load(str(MODEL_PATH))
prediction_lock = Lock()

logger.info("Spark %s started", spark.version)
logger.info("PipelineModel loaded from %s", MODEL_PATH)


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


INCIDENT_SCHEMA = StructType(
    [
        StructField("country", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("attack_type", StringType(), False),
        StructField("target_industry", StringType(), False),
        StructField("financial_loss_million", DoubleType(), False),
        StructField("affected_users", LongType(), False),
        StructField("attack_source", StringType(), False),
        StructField("vulnerability_type", StringType(), False),
        StructField("defense_mechanism", StringType(), False),
    ]
)


app = FastAPI(
    title="Cyber Incident Resolution-Time API",
    description=(
        "Loads the Spark Random Forest PipelineModel trained in Part I "
        "and predicts incident resolution time in hours."
    ),
    version="1.0.0",
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
        "model_type": "Spark ML Random Forest PipelineModel",
        "spark_version": spark.version,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CyberIncidentInput) -> PredictionOutput:
    """Predict resolution time after recreating Part I engineered features."""
    incident_row = [
        (
            data.country,
            data.year,
            data.attack_type,
            data.target_industry,
            float(data.financial_loss_million),
            int(data.affected_users),
            data.attack_source,
            data.vulnerability_type,
            data.defense_mechanism,
        )
    ]

    try:
        incident_df = spark.createDataFrame(
            incident_row,
            schema=INCIDENT_SCHEMA,
        )

        engineered_df = (
            incident_df
            .withColumn(
                "years_since_2015",
                F.col("year") - F.lit(2015),
            )
            .withColumn(
                "loss_per_user_usd",
                (
                    F.col("financial_loss_million")
                    * F.lit(1_000_000.0)
                ) / F.col("affected_users"),
            )
        )

        # A single worker and lock avoid concurrent access to the local JVM.
        with prediction_lock:
            result = (
                model.transform(engineered_df)
                .select("predicted_resolution_time")
                .first()
            )

        if result is None:
            raise RuntimeError("The Spark model returned no prediction.")

        predicted_hours = max(
            0.0,
            float(result["predicted_resolution_time"]),
        )

        return PredictionOutput(
            predicted_resolution_time_hours=round(predicted_hours, 2),
            predicted_resolution_time_days=round(predicted_hours / 24, 2),
            model_type="Spark ML Random Forest Regressor",
            spark_version=spark.version,
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc
