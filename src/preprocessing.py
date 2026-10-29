import numpy as np
import pandas as pd
from scipy.stats import yeojohnson


# Exact feature order expected by the clustering model
MODEL_FEATURES = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp",
]


# Lambda learned from the original training dataset
INFLATION_YJ_LAMBDA = 0.4278346215723399


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the feature transformations used during model training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw country-level input data.

    Returns
    -------
    pd.DataFrame
        Transformed features in the exact order expected by the scaler
        and clustering model.
    """

    transformed = df[MODEL_FEATURES].copy()

    # Reduce right-skewed distributions
    transformed["child_mort"] = np.log1p(
        transformed["child_mort"]
    )

    transformed["gdpp"] = np.log1p(
        transformed["gdpp"]
    )

    transformed["exports"] = np.sqrt(
        transformed["exports"]
    )

    transformed["imports"] = np.sqrt(
        transformed["imports"]
    )

    # Reuse the lambda learned during training.
    # Do NOT estimate a new lambda for incoming observations.
    transformed["inflation"] = yeojohnson(
        transformed["inflation"],
        lmbda=INFLATION_YJ_LAMBDA,
    )

    return transformed