import pandas as pd
import numpy as np

from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.combine import SMOTETomek


def prepare_raw_data(
        raw_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Initial data preprocessing.

    Operations:
    - grouping education categories;
    - creation of additional features;
    - processing special values in pdays.
    """

    df = raw_df.copy()

    education_map = {
        'basic.4y': 'basic.education',
        'basic.6y': 'basic.education',
        'basic.9y': 'basic.education',
        'illiterate': 'basic.education'
    }

    df['education'] = df['education'].replace(education_map)

    df['pdays_contacted'] = (
            df['pdays'] != 999
    ).astype(int)

    df['pdays'] = df['pdays'].replace(999, -1)
    return df


def split_data(
        raw_df: pd.DataFrame,
        target_col: str = "y",
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Split data into train, validation and test sets.

    Test = 20%
    Validation = 20% of the original dataset
    Train = 60%

    Stratification is used because the target is imbalanced.
    """

    train_val_df, test_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col]
    )

    val_relative_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_df[target_col]
    )

    return train_df, val_df, test_df


def create_inputs_targets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        input_cols: List[str],
        target_col: str
):
    """
    Separate features and target.
    """

    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    X_test = test_df[input_cols].copy()
    y_test = test_df[target_col].copy()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


def encode_target(
        y_train,
        y_val,
        y_test
):
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)

    y_val = encoder.transform(y_val)

    y_test = encoder.transform(y_test)

    return y_train, y_val, y_test, encoder


def encode_binary_features(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
):
    """
    Encode binary categorical features.

    Unknown values are encoded as -1.
    """

    mappings = {

        'default': {
            'no': 0,
            'yes': 1,
            'unknown': -1
        },

        'housing': {
            'no': 0,
            'yes': 1,
            'unknown': -1
        },

        'loan': {
            'no': 0,
            'yes': 1,
            'unknown': -1
        }
    }

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for col, mapping in mappings.items():
        X_train[col] = X_train[col].map(mapping).fillna(-1)
        X_val[col] = X_val[col].map(mapping).fillna(-1)
        X_test[col] = X_test[col].map(mapping).fillna(-1)

    return X_train, X_val, X_test


def impute_missing_values(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
):
    """
    Fill missing values.

    Numeric -> median calculated only on train.
    Categorical -> most frequent value calculated only on train.
    """

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    if numeric_cols:
        X_train[numeric_cols] = numeric_imputer.fit_transform(
            X_train[numeric_cols]
        )

        X_val[numeric_cols] = numeric_imputer.transform(
            X_val[numeric_cols]
        )

        X_test[numeric_cols] = numeric_imputer.transform(
            X_test[numeric_cols]
        )

    if categorical_cols:
        X_train[categorical_cols] = categorical_imputer.fit_transform(
            X_train[categorical_cols]
        )

        X_val[categorical_cols] = categorical_imputer.transform(
            X_val[categorical_cols]
        )

        X_test[categorical_cols] = categorical_imputer.transform(
            X_test[categorical_cols]
        )

    return (
        X_train,
        X_val,
        X_test,
        numeric_imputer,
        categorical_imputer
    )


def scale_numeric_features(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_cols: List[str]
):
    """
    Scale numeric features using StandardScaler.

    """

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(
            X_train[numeric_cols]
        )

        X_val[numeric_cols] = scaler.transform(
            X_val[numeric_cols]
        )

        X_test[numeric_cols] = scaler.transform(
            X_test[numeric_cols]
        )

    return X_train, X_val, X_test, scaler


def encode_categorical_features(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        categorical_cols: List[str]
):
    """
    One-hot encode categorical features.

    """

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    train_encoded = encoder.fit_transform(
        X_train[categorical_cols]
    )

    val_encoded = encoder.transform(
        X_val[categorical_cols]
    )

    test_encoded = encoder.transform(
        X_test[categorical_cols]
    )

    encoded_cols = encoder.get_feature_names_out(
        categorical_cols
    )

    train_encoded = pd.DataFrame(
        train_encoded,
        columns=encoded_cols,
        index=X_train.index
    )

    val_encoded = pd.DataFrame(
        val_encoded,
        columns=encoded_cols,
        index=X_val.index
    )

    test_encoded = pd.DataFrame(
        test_encoded,
        columns=encoded_cols,
        index=X_test.index
    )

    X_train = pd.concat(
        [
            X_train.drop(columns=categorical_cols),
            train_encoded
        ],
        axis=1
    )

    X_val = pd.concat(
        [
            X_val.drop(columns=categorical_cols),
            val_encoded
        ],
        axis=1
    )

    X_test = pd.concat(
        [
            X_test.drop(columns=categorical_cols),
            test_encoded
        ],
        axis=1
    )

    return X_train, X_val, X_test, encoder


def encode_categorical_features_ordinal(
        X_train,
        X_val,
        X_test,
        categorical_cols
):
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    X_train[categorical_cols] = encoder.fit_transform(
        X_train[categorical_cols]
    )

    X_val[categorical_cols] = encoder.transform(
        X_val[categorical_cols]
    )

    X_test[categorical_cols] = encoder.transform(
        X_test[categorical_cols]
    )

    return X_train, X_val, X_test, encoder


def detect_outliers_iqr(
        df: pd.DataFrame,
        numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Detect outliers using IQR method.

    Returns a DataFrame containing the number of outliers
    for each numeric feature.

    This function only detects outliers.
    It does not automatically remove them.
    """

    results = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (
                (df[col] < lower_bound) |
                (df[col] > upper_bound)
        )

        results.append({
            'feature': col,
            'outliers_count': outliers.sum(),
            'outliers_percent':
                round(outliers.mean() * 100, 2),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

    return pd.DataFrame(results)


def clip_outliers(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_cols: List[str]
):
    """
    Clip extreme values using IQR boundaries.

    Boundaries are calculated only on train.
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    bounds = {}

    for col in numeric_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        bounds[col] = (lower, upper)

        X_train[col] = X_train[col].clip(
            lower=lower,
            upper=upper
        )

        X_val[col] = X_val[col].clip(
            lower=lower,
            upper=upper
        )

        X_test[col] = X_test[col].clip(
            lower=lower,
            upper=upper
        )

    return X_train, X_val, X_test, bounds


def preprocess_data(
        raw_df: pd.DataFrame,
        scale_numeric: bool = True,
        handle_outliers: bool = False,
        one_hot: bool = True
):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw dataset.

    scale_numeric : bool
        Whether to scale numerical features.

    handle_outliers : bool
        Whether to clip outliers using IQR.

    one_hot : bool
        If True, categorical features are encoded using OneHotEncoder.
        If False, categorical features are encoded using OrdinalEncoder.
    """

    target_col = "y"

    # --------------------------------------------------------
    # 1. Initial preprocessing
    # --------------------------------------------------------

    df = prepare_raw_data(raw_df)

    # --------------------------------------------------------
    # 2. Columns
    # --------------------------------------------------------

    input_cols = df.columns.drop(
        [target_col]
    ).tolist()

    # --------------------------------------------------------
    # 3. Train / validation / test
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df
    ) = split_data(
        df,
        target_col=target_col
    )

    # --------------------------------------------------------
    # 4. X / y
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = create_inputs_targets(
        train_df,
        val_df,
        test_df,
        input_cols,
        target_col
    )

    # --------------------------------------------------------
    # 5. Target encoding
    # --------------------------------------------------------

    (
        y_train,
        y_val,
        y_test,
        target_encoder
    ) = encode_target(
        y_train,
        y_val,
        y_test
    )

    # --------------------------------------------------------
    # 6. Feature types
    # --------------------------------------------------------

    binary_cols = [
        'default',
        'housing',
        'loan'
    ]

    categorical_cols = [
        'job',
        'marital',
        'education',
        'contact',
        'month',
        'day_of_week',
        'poutcome'
    ]

    numeric_cols = [
        col for col in X_train.columns
        if col not in binary_cols + categorical_cols
    ]

    # --------------------------------------------------------
    # 7. Binary encoding
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = encode_binary_features(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # 8. Outlier handling
    # --------------------------------------------------------

    outlier_report = detect_outliers_iqr(
        X_train,
        numeric_cols
    )

    if handle_outliers:

        (
            X_train,
            X_val,
            X_test,
            outlier_bounds
        ) = clip_outliers(
            X_train,
            X_val,
            X_test,
            numeric_cols
        )

    else:

        outlier_bounds = None

    # --------------------------------------------------------
    # 9. Missing values
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        numeric_imputer,
        categorical_imputer
    ) = impute_missing_values(
        X_train,
        X_val,
        X_test,
        numeric_cols,
        categorical_cols
    )

    # --------------------------------------------------------
    # 10. Scaling
    # --------------------------------------------------------

    scaler = None

    if scale_numeric:
        (
            X_train,
            X_val,
            X_test,
            scaler
        ) = scale_numeric_features(
            X_train,
            X_val,
            X_test,
            numeric_cols
        )

    # --------------------------------------------------------
    # 11. Categorical encoding
    # --------------------------------------------------------

    encoder = None

    if one_hot:

        (
            X_train,
            X_val,
            X_test,
            encoder
        ) = encode_categorical_features(
            X_train,
            X_val,
            X_test,
            categorical_cols
        )

    else:

        (
            X_train,
            X_val,
            X_test,
            encoder
        ) = encode_categorical_features_ordinal(
            X_train,
            X_val,
            X_test,
            categorical_cols
        )

    # --------------------------------------------------------
    # 12. Return
    # --------------------------------------------------------

    return {
        'X_train': X_train,
        'y_train': y_train,

        'X_val': X_val,
        'y_val': y_val,

        'X_test': X_test,
        'y_test': y_test,

        'input_cols': input_cols,

        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'binary_cols': binary_cols,

        'target_encoder': target_encoder,

        'scaler': scaler,
        'encoder': encoder,

        'numeric_imputer': numeric_imputer,
        'categorical_imputer': categorical_imputer,

        'outlier_report': outlier_report,
        'outlier_bounds': outlier_bounds
    }


#  Preprocessing NEW data

def preprocess_new_data(
        new_df: pd.DataFrame,
        input_cols: List[str],
        numeric_cols: List[str],
        categorical_cols: List[str],
        scaler: Optional[StandardScaler],
        encoder: OneHotEncoder,
        numeric_imputer: SimpleImputer,
        categorical_imputer: SimpleImputer,
        scale_numeric: bool = True
):
    """
    Preprocess completely new data using transformers
    fitted on training data.

    IMPORTANT:
    No fit() is performed here.
    Only transform() is used.
    """

    X = new_df.copy()

    education_map = {
        'basic.4y': 'basic.education',
        'basic.6y': 'basic.education',
        'basic.9y': 'basic.education',
        'illiterate': 'basic.education'
    }

    X['education'] = X['education'].replace(
        education_map
    )

    X['pdays_contacted'] = (
            X['pdays'] != 999
    ).astype(int)

    X['pdays'] = X['pdays'].replace(
        999,
        -1
    )

    X = X[input_cols].copy()

    mappings = {

        'default': {
            'no': 0,
            'yes': 1,
            'unknown': -1
        },

        'housing': {
            'no': 0,
            'yes': 1,
            'unknown': -1
        },

        'loan': {
            'no': 0,
            'yes': 1,
            'unknown': -1
        }
    }

    for col, mapping in mappings.items():
        X[col] = X[col].map(
            mapping
        ).fillna(-1)

    if numeric_cols:
        X[numeric_cols] = numeric_imputer.transform(
            X[numeric_cols]
        )

    if categorical_cols:
        X[categorical_cols] = categorical_imputer.transform(
            X[categorical_cols]
        )

    if scale_numeric and scaler is not None:
        X[numeric_cols] = scaler.transform(
            X[numeric_cols]
        )

    encoded = encoder.transform(
        X[categorical_cols]
    )

    encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_cols
        ),
        index=X.index
    )

    X = pd.concat(
        [
            X.drop(columns=categorical_cols),
            encoded
        ],
        axis=1
    )

    return X


def apply_smotenc(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: list,
        random_state: int = 42
):
    """
    Apply SMOTENC only to the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    categorical_cols : list
        Names of categorical columns.

    random_state : int
        Random seed.

    Returns
    -------
    X_resampled : pd.DataFrame
        Resampled training features.

    y_resampled : pd.Series
        Resampled training target.
    """

    cat_feature_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_cols
    ]

    smotenc = SMOTENC(
        categorical_features=cat_feature_indices,
        random_state=random_state
    )

    X_resampled, y_resampled = smotenc.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


def apply_smotetomek(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: list,
        random_state: int = 42
):
    """
    Apply SMOTENC + Tomek Links only to the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    categorical_cols : list
        Names of categorical columns.

    random_state : int
        Random seed.

    Returns
    -------
    X_resampled : pd.DataFrame
        Resampled training features.

    y_resampled : pd.Series
        Resampled training target.
    """

    cat_feature_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_cols
    ]

    smote = SMOTENC(
        categorical_features=cat_feature_indices,
        random_state=random_state
    )

    smote_tomek = SMOTETomek(
        smote=smote,
        random_state=random_state
    )

    X_resampled, y_resampled = smote_tomek.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled