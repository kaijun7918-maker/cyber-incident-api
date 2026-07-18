"""FastAPI service for the exported Spark Random Forest model."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

import joblib

from fastapi import (
    FastAPI,
    HTTPException
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)



# Application logging


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    "cyber-incident-api"
)



# Locate the lightweight Joblib deployment model


BASE_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_PATH = (
    BASE_DIRECTORY /
    "model" /
    "model.joblib"
)



# Lightweight Spark Random Forest predictor


class PortableSparkRandomForest:
    """
    Apply the preprocessing and decision-tree rules exported
    from the original Apache Spark ML PipelineModel.

    Spark and Java are not required during prediction.
    """

    def __init__(
        self,
        model_path: Path
    ) -> None:

        if not model_path.is_file():
            raise RuntimeError(
                "Joblib model was not found at: "
                f"{model_path}"
            )

        artifact = joblib.load(
            model_path
        )

        if not isinstance(
            artifact,
            dict
        ):
            raise RuntimeError(
                "The Joblib artifact must "
                "contain a dictionary."
            )

        if (
            artifact.get(
                "format_version"
            ) != 1
        ):
            raise RuntimeError(
                "Unsupported Joblib model format."
            )

        if (
            artifact.get(
                "model_type"
            ) != "RandomForestRegressor"
        ):
            raise RuntimeError(
                "The Joblib artifact is not "
                "a Random Forest regression model."
            )

        self.artifact = artifact

        self.model_name = artifact[
            "model_name"
        ]

        self.source_model = artifact[
            "source_model"
        ]

        self.spark_version = artifact[
            "spark_version"
        ]

        self.number_of_trees = int(
            artifact[
                "number_of_trees"
            ]
        )

        self.number_of_features = int(
            artifact[
                "number_of_features"
            ]
        )

        self.string_indexers = artifact[
            "string_indexers"
        ]

        self.encoder = artifact[
            "one_hot_encoder"
        ]

        self.assembler_columns = artifact[
            "assembler_input_columns"
        ]

        self.trees = artifact[
            "trees"
        ]

        self.tree_weights = [
            float(weight)

            for weight in artifact[
                "tree_weights"
            ]
        ]

        self.feature_engineering = (
            artifact.get(
                "feature_engineering",
                {
                    "year_base": 2015,
                    "financial_loss_multiplier":
                        1_000_000.0
                }
            )
        )

        self.indexer_by_output = {
            indexer[
                "output_column"
            ]: indexer

            for indexer
            in self.string_indexers
        }

        self.label_mappings = {
            indexer[
                "output_column"
            ]: {
                label: position

                for position, label
                in enumerate(
                    indexer[
                        "labels"
                    ]
                )
            }

            for indexer
            in self.string_indexers
        }

        self.total_tree_weight = sum(
            self.tree_weights
        )

        self._validate_artifact()


    def _validate_artifact(
        self
    ) -> None:
        """Validate the exported model structure."""

        if (
            len(self.trees)
            != self.number_of_trees
        ):
            raise RuntimeError(
                "Tree-count mismatch in "
                "the Joblib artifact."
            )

        if (
            len(self.tree_weights)
            != self.number_of_trees
        ):
            raise RuntimeError(
                "Tree-weight count does not "
                "match the number of trees."
            )

        if self.total_tree_weight <= 0:
            raise RuntimeError(
                "The total tree weight "
                "must be greater than zero."
            )

        required_encoder_fields = {
            "input_columns",
            "output_columns",
            "encoded_vector_sizes"
        }

        if not required_encoder_fields.issubset(
            self.encoder
        ):
            raise RuntimeError(
                "The OneHotEncoder configuration "
                "is incomplete."
            )

        encoder_lengths = {
            len(
                self.encoder[
                    "input_columns"
                ]
            ),
            len(
                self.encoder[
                    "output_columns"
                ]
            ),
            len(
                self.encoder[
                    "encoded_vector_sizes"
                ]
            )
        }

        if len(encoder_lengths) != 1:
            raise RuntimeError(
                "The OneHotEncoder configuration "
                "contains inconsistent lengths."
            )

        logger.info(
            "Loaded %s with %s trees and %s features.",
            self.model_name,
            self.number_of_trees,
            self.number_of_features
        )


    def build_feature_vector(
        self,
        incident: dict[str, Any]
    ) -> list[float]:
        """
        Recreate feature engineering, StringIndexer,
        OneHotEncoder and VectorAssembler operations.
        """

        prepared_incident = dict(
            incident
        )

        year_base = int(
            self.feature_engineering.get(
                "year_base",
                2015
            )
        )

        financial_multiplier = float(
            self.feature_engineering.get(
                "financial_loss_multiplier",
                1_000_000.0
            )
        )

        prepared_incident[
            "years_since_2015"
        ] = (
            int(
                prepared_incident[
                    "year"
                ]
            ) -
            year_base
        )

        prepared_incident[
            "loss_per_user_usd"
        ] = (
            float(
                prepared_incident[
                    "financial_loss_million"
                ]
            ) *
            financial_multiplier /
            int(
                prepared_incident[
                    "affected_users"
                ]
            )
        )

        encoded_columns: dict[
            str,
            list[float]
        ] = {}

        for (
            index_column,
            encoded_column,
            encoded_size
        ) in zip(
            self.encoder[
                "input_columns"
            ],
            self.encoder[
                "output_columns"
            ],
            self.encoder[
                "encoded_vector_sizes"
            ]
        ):

            indexer = (
                self.indexer_by_output[
                    index_column
                ]
            )

            raw_column = indexer[
                "input_column"
            ]

            raw_value = str(
                prepared_incident[
                    raw_column
                ]
            ).strip()

            label_mapping = (
                self.label_mappings[
                    index_column
                ]
            )

            # Unknown categories use the extra category created
            # by StringIndexer(handleInvalid="keep").
            category_index = (
                label_mapping.get(
                    raw_value,
                    len(
                        indexer[
                            "labels"
                        ]
                    )
                )
            )

            encoded_vector = [
                0.0
            ] * int(
                encoded_size
            )

            # If dropLast is enabled, the final category
            # is represented using an all-zero vector.
            if (
                0 <= category_index <
                len(encoded_vector)
            ):
                encoded_vector[
                    category_index
                ] = 1.0

            encoded_columns[
                encoded_column
            ] = encoded_vector

        feature_vector: list[
            float
        ] = []

        for column_name in (
            self.assembler_columns
        ):

            if (
                column_name
                in encoded_columns
            ):
                feature_vector.extend(
                    encoded_columns[
                        column_name
                    ]
                )

            elif (
                column_name
                in prepared_incident
            ):
                feature_vector.append(
                    float(
                        prepared_incident[
                            column_name
                        ]
                    )
                )

            else:
                raise ValueError(
                    "Required model feature is missing: "
                    f"{column_name}"
                )

        if (
            len(feature_vector)
            != self.number_of_features
        ):
            raise ValueError(
                "Feature-vector length mismatch. "
                f"Expected {self.number_of_features}, "
                f"received {len(feature_vector)}."
            )

        return feature_vector


    @staticmethod
    def predict_tree(
        tree: dict[str, Any],
        features: list[float]
    ) -> float:
        """Generate a prediction from one decision tree."""

        current_node = tree

        while (
            current_node[
                "node_type"
            ] != "leaf"
        ):

            feature_index = int(
                current_node[
                    "feature_index"
                ]
            )

            feature_value = features[
                feature_index
            ]

            if (
                current_node[
                    "split_type"
                ] == "continuous"
            ):
                move_left = (
                    feature_value <=
                    float(
                        current_node[
                            "threshold"
                        ]
                    )
                )

            elif (
                current_node[
                    "split_type"
                ] == "categorical"
            ):
                move_left = (
                    feature_value in
                    current_node[
                        "left_categories"
                    ]
                )

            else:
                raise ValueError(
                    "Unsupported tree split type: "
                    f"{current_node['split_type']}"
                )

            current_node = (
                current_node[
                    "left_child"
                ]
                if move_left
                else
                current_node[
                    "right_child"
                ]
            )

        return float(
            current_node[
                "prediction"
            ]
        )


    def predict(
        self,
        incident: dict[str, Any]
    ) -> float:
        """Generate the weighted Random Forest prediction."""

        features = (
            self.build_feature_vector(
                incident
            )
        )

        weighted_prediction = sum(
            weight *
            self.predict_tree(
                tree,
                features
            )

            for tree, weight in zip(
                self.trees,
                self.tree_weights
            )
        )

        prediction = (
            weighted_prediction /
            self.total_tree_weight
        )

        return max(
            0.0,
            float(prediction)
        )


    def get_categories(
        self
    ) -> dict[str, list[str]]:
        """Return known categorical values for the UI."""

        return {
            indexer[
                "input_column"
            ]: list(
                indexer[
                    "labels"
                ]
            )

            for indexer
            in self.string_indexers
        }



# Load the deployment model once when the API starts


model = PortableSparkRandomForest(
    MODEL_PATH
)



# FastAPI input and output schemas


class CyberIncidentInput(
    BaseModel
):
    """Incident information accepted by the API."""

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
                "defense_mechanism": "Firewall"
            }
        }
    )

    country: str = Field(
        min_length=1,
        max_length=100
    )

    year: int = Field(
        ge=2015,
        le=2024
    )

    attack_type: str = Field(
        min_length=1,
        max_length=100
    )

    target_industry: str = Field(
        min_length=1,
        max_length=100
    )

    financial_loss_million: float = Field(
        ge=0
    )

    affected_users: int = Field(
        gt=0
    )

    attack_source: str = Field(
        min_length=1,
        max_length=100
    )

    vulnerability_type: str = Field(
        min_length=1,
        max_length=100
    )

    defense_mechanism: str = Field(
        min_length=1,
        max_length=100
    )

    @field_validator(
        "country",
        "attack_type",
        "target_industry",
        "attack_source",
        "vulnerability_type",
        "defense_mechanism",
        mode="before"
    )
    @classmethod
    def strip_text_values(
        cls,
        value: object
    ) -> object:
        """Remove leading and trailing spaces."""

        if isinstance(
            value,
            str
        ):
            return value.strip()

        return value


class PredictionOutput(
    BaseModel
):
    """Prediction response returned by the API."""

    predicted_resolution_time_hours: float

    predicted_resolution_time_days: float

    model_type: str

    source_model: str



# Create the FastAPI application


app = FastAPI(
    title=(
        "Cyber Incident "
        "Resolution-Time API"
    ),

    description=(
        "Serves the Random Forest model trained "
        "with Apache Spark MLlib in Part I. "
        "The fitted tree structure was exported "
        "as a lightweight Joblib artifact for "
        "cloud deployment."
    ),

    version="2.0.0"
)


# Allow the Streamlit dashboard to access the API

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST"
    ],

    allow_headers=[
        "*"
    ]
)



# API routes


@app.get("/")
def root() -> dict[
    str,
    str
]:
    """Return basic API information."""

    return {
        "message": (
            "Cyber Incident Resolution-Time "
            "API is running."
        ),

        "documentation": "/docs",

        "health": "/health",

        "metadata": "/metadata",

        "prediction": "/predict"
    }


@app.get("/health")
def health() -> dict[
    str,
    Any
]:
    """Confirm that the deployment model is available."""

    return {
        "status": "healthy",

        "model_loaded": True,

        "model_type": (
            "Portable Spark ML "
            "Random Forest Regressor"
        ),

        "source_model":
            model.source_model,

        "number_of_trees":
            model.number_of_trees,

        "number_of_features":
            model.number_of_features,

        "spark_training_version":
            model.spark_version
    }


@app.get("/metadata")
def metadata() -> dict[
    str,
    Any
]:
    """Return model categories for the Streamlit form."""

    return {
        "model_name":
            model.model_name,

        "source_model":
            model.source_model,

        "number_of_trees":
            model.number_of_trees,

        "number_of_features":
            model.number_of_features,

        "categories":
            model.get_categories()
    }


@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    data: CyberIncidentInput
) -> PredictionOutput:
    """Predict incident resolution time."""

    try:
        predicted_hours = model.predict(
            data.model_dump()
        )

        return PredictionOutput(
            predicted_resolution_time_hours=round(
                predicted_hours,
                2
            ),

            predicted_resolution_time_days=round(
                predicted_hours / 24,
                2
            ),

            model_type=(
                "Portable Spark ML "
                "Random Forest Regressor"
            ),

            source_model=(
                model.source_model
            )
        )

    except Exception as exception:
        logger.exception(
            "Prediction failed: %s",
            exception
        )

        raise HTTPException(
            status_code=500,

            detail=(
                "Prediction failed. "
                "Check the server logs."
            )
        ) from exception