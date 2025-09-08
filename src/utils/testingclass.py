# ===========================
# GhanaRainfallPredictor (Leak-free, Macro-F1 aware)
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, make_scorer
)
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, TimeSeriesSplit,
    cross_validate, train_test_split
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import FunctionTransformer
import warnings
from collections import Counter



# ---------- Utility: version-proof OHE ----------
def _make_ohe():
    """Return an OneHotEncoder that works across sklearn versions."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------- Custom transformers (stateless or fold-safe) ----------
class TimeFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create deterministic time features from 'prediction_time' (no learned state)."""
    def __init__(self, time_col: str = "prediction_time"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.time_col in X.columns:
            dt = pd.to_datetime(X[self.time_col], errors="coerce")
            X["hour"] = dt.dt.hour
            X["day_of_week"] = dt.dt.dayofweek   # 0=Mon
            X["month"] = dt.dt.month
            X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)
            X["is_morning"] = X["hour"].between(6, 11, inclusive="both").astype(int)
            X["is_afternoon"] = X["hour"].between(12, 17, inclusive="both").astype(int)
            X["is_evening"] = X["hour"].between(18, 21, inclusive="both").astype(int)
            X["is_night"] = ((X["hour"] >= 22) | (X["hour"] < 6)).astype(int)
        return X


class SimpleFeatureFixes(BaseEstimator, TransformerMixin):
    """Simple imputations/flags that don’t require fitting."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "indicator" in X.columns:
            X["indicator"] = X["indicator"].fillna("no_indicator")
            X["has_indicator"] = (X["indicator"] != "no_indicator").astype(int)
        return X


class GroupAggregates(BaseEstimator, TransformerMixin):
    def __init__(self, agg_specs=None, group_cols=None):
        # Store parameters exactly as passed (don't modify them)
        self.agg_specs = agg_specs
        self.group_cols = group_cols
    
    def fit(self, X, y=None):
        # Validation and setup can happen in fit
        if self.agg_specs is None:
            self.agg_specs_ = []
        else:
            self.agg_specs_ = list(self.agg_specs)  # Create a copy for internal use
            
        if self.group_cols is None:
            self.group_cols_ = []
        else:
            self.group_cols_ = list(self.group_cols)  # Create a copy for internal use
        
        return self
    
    def transform(self, X):
        # Check if we have the required columns
        if not self.group_cols_:
            return X
            
        missing_cols = [col for col in self.group_cols_ if col not in X.columns]
        if missing_cols:
            print(f"Warning: GroupAggregates missing columns {missing_cols}, returning original data")
            return X
        
        X_new = X.copy()
        
        # Apply aggregations
        for col, agg_func in self.agg_specs_:
            if col in X.columns:
                try:
                    agg_result = X.groupby(self.group_cols_)[col].transform(agg_func)
                    new_col_name = f"{col}_{agg_func}_by_{'_'.join(self.group_cols_)}"
                    X_new[new_col_name] = agg_result
                except Exception as e:
                    print(f"Warning: Failed to aggregate {col} with {agg_func}: {e}")
        
        return X_new
    
    def get_params(self, deep=True):
        # This is crucial for sklearn compatibility
        return {'agg_specs': self.agg_specs, 'group_cols': self.group_cols}
    
    def set_params(self, **params):
        # This is also crucial for sklearn compatibility
        for param, value in params.items():
            setattr(self, param, value)
        return self

# ---------- Main predictor ----------
class GhanaRainfallPredictor:
    """
    End-to-end, leak-free training & evaluation with Macro-F1 and robust CV.
    Usage:
        pred = GhanaRainfallPredictor(use_smote=True, group_cv_col="community")
        df = pred.load_and_explore_dta("train.csv")
        X, y = pred.prepare_features_and_target(df, "Target")
        pred.evaluate_cv(X, y)      # CV Accuracy + Macro-F1 (robust)
        pred.train_holdout(X, y)    # Fit final pipeline + holdout metrics
        pred.test_on_file("test.csv", target_column="Target")
        pred.save_submission("test.csv", "SampleSubmission.csv", "MySubmission.csv")
    """
    def __init__(self, use_smote: bool = True, group_cv_col: Optional[str] = None, time_cv_col: Optional[str] = None):
        self.use_smote = use_smote
        self.group_cv_col = group_cv_col
        self.time_cv_col = time_cv_col

        self.label_encoder = None
        self.final_model = None       # fitted Pipeline
        self.performance_metrics = {}

        # feature name intents (only used if present)
        self.categorical_features = ["indicator", "community", "district", "day_of_week", "month"]
        self.numerical_features = [
            "confidence", "predicted_intensity", "forecast_length",
            "hour", "is_weekend", "is_morning", "is_afternoon", "is_evening", "is_night",
            # aggregates created by GroupAggregates (if groups exist)
            "community_confidence_mean", "district_confidence_mean",
            "community_confidence_median", "district_confidence_median",
            # keep any pre-engineered columns if present
            "community_avg_confidence", "district_avg_confidence"
        ]

    # --------- EDA loader (inspection only; not used by the pipeline) ----------
    def load_and_explore_dta(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print("Dataset loaded Successfully")
            print(f"Shape: {df.shape}")
            print("\nColumns:", df.columns.tolist())
            print(df.head())

            print("\nDataset Info:")
            print(df.info())

            print("\nMissing Values (top 10):")
            mv = df.isnull().sum().sort_values(ascending=False)
            mv_pct = (mv / len(df) * 100).round(2)
            print(pd.DataFrame({"Missing Count": mv, "Missing %": mv_pct}).head(10))

            if "Target" in df.columns:
                print("\nTARGET VARIABLE ANALYSIS")
                counts = df["Target"].value_counts()
                print(counts, "\n")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    # --------- Model pieces ----------
    def build_ensemble_model(self):
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            random_state=42, class_weight="balanced_subsample", n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        )
        svm = SVC(
            kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42
        )
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft', n_jobs=-1
        )

    def create_group_aggregates_func(agg_specs, group_cols):
        """Create a function that can be used with FunctionTransformer"""
        def group_aggregates_transform(X):
            if not group_cols or not agg_specs:
                return X
            
            missing_cols = [col for col in group_cols if col not in X.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}, returning original data")
                return X
        
            X_new = X.copy()
        
            for col, agg_func in agg_specs:
                if col in X.columns:
                    try:
                        agg_result = X.groupby(group_cols)[col].transform(agg_func)
                        new_col_name = f"{col}_{agg_func}_by_{'_'.join(group_cols)}"
                        X_new[new_col_name] = agg_result
                    except Exception as e:
                        print(f"Warning: Failed to aggregate {col} with {agg_func}: {e}")
        
            return X_new
    
        return group_aggregates_transform
    
    def build_pipeline(self, X_cols: List[str]):
        cats = [c for c in self.categorical_features if c in X_cols]
        nums = [n for n in self.numerical_features if n in X_cols]

        group_agg_func = self.create_group_aggregates_func(
        agg_specs=[('confidence', 'mean'), ('confidence', 'median')],
        group_cols=['community', 'district']
        )

        group_agg_transformer = FunctionTransformer(
            group_agg_func,
            validate=False,  # Important for pandas DataFrames
            check_inverse=False
        )

        ohe = _make_ohe()
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [n for n in nums if n in X_cols]),
                ("cat", ohe, [c for c in cats if c in X_cols]),
            ],
            remainder="drop"
        )

        steps = [
            ("time", TimeFeatureBuilder("prediction_time")),
            ("fix", SimpleFeatureFixes()),
            ("grp", grpagg),
            ("pre", pre),
        ]
        if self.use_smote:
            steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("clf", self.build_ensemble_model()))
        return ImbPipeline(steps)

    # --------- Features/labels ----------
    def prepare_features_and_target(self, df: pd.DataFrame, target_column: str = "Target"):
        exclude = [target_column, "ID", "prediction_time", "user_id"]
        X = df[[c for c in df.columns if c not in exclude]].copy()
        y = df[target_column].values
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        print("Target classes:", list(self.label_encoder.classes_))
        return X, y_enc

    def evaluate_cv(self, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5):
        # Pre-apply GroupAggregates transformation outside of pipeline
        group_agg = GroupAggregates(
        agg_specs=[('confidence', 'mean'), ('confidence', 'median')], 
        group_cols=['community', 'district']
        )
    
        try:
        # Apply the transformation once
            X_transformed = group_agg.fit_transform(X)
        except Exception as e:
            print(f"Warning: GroupAggregates failed, using original data: {e}")
            X_transformed = X
    
        # Build pipeline WITHOUT the problematic GroupAggregates step
        pipe = self.build_pipeline(list(X_transformed.columns))

        # Auto-adjust splits so every class has at least n_splits members
        class_counts = Counter(y)
        min_class = min(class_counts.values())
        effective_splits = max(2, min(n_splits, min_class))

        # Choose splitter
        if self.group_cv_col and (self.group_cv_col in X_transformed.columns):
            gkf = GroupKFold(n_splits=effective_splits)
            cv_iter = gkf.split(X_transformed, y, groups=X_transformed[self.group_cv_col])
        elif self.time_cv_col and (self.time_cv_col in X_transformed.columns):
            cv_iter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42).split(X_transformed, y)
        else:
            cv_iter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42).split(X_transformed, y)

        # Cross-validate with transformed data
        scores = cross_validate(
            pipe, X_transformed, y, cv=cv_iter, n_jobs=1,
            scoring={"acc": "accuracy", "f1m": make_scorer(f1_score, average="macro")}
        )
    
        acc_mean, acc_ci = scores["test_acc"].mean(), scores["test_acc"].std()*2
        f1m_mean, f1m_ci = scores["test_f1m"].mean(), scores["test_f1m"].std()*2
        print(f"CV folds: {effective_splits}")
        print(f"CV Accuracy: {acc_mean:.4f} ± {acc_ci:.4f}")
        print(f"CV Macro-F1: {f1m_mean:.4f} ± {f1m_ci:.4f}")
        self.performance_metrics.update({"cv_acc": acc_mean, "cv_f1_macro": f1m_mean})
        return scores
    # --------- Train with holdout (prints both metrics) ----------
    def train_holdout(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        pipe = self.build_pipeline(list(X.columns))
        pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        print("\n=== Holdout Performance ===")
        print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_te, y_pred, target_names=self.label_encoder.classes_))
        print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))

        self.final_model = pipe
        self.performance_metrics.update({"holdout_acc": acc, "holdout_f1_macro": f1m})
        return pipe

    # --------- Test on external file (prints metrics if labels exist) ----------
    def test_on_file(self, test_file: str, target_column: Optional[str] = "Target"):
        if self.final_model is None:
            raise ValueError("Train the model first (final_model is None).")

        test_df = pd.read_csv(test_file)
        exclude = ["ID", "prediction_time", "user_id"]
        y_true_enc = None
        if target_column and (target_column in test_df.columns):
            if self.label_encoder is None:
                raise ValueError("LabelEncoder not fitted. Train first.")
            y_true_enc = self.label_encoder.transform(test_df[target_column])
            exclude.append(target_column)

        X_test = test_df[[c for c in test_df.columns if c not in exclude]].copy()
        y_pred = self.final_model.predict(X_test)
        y_prob = self.final_model.predict_proba(X_test)
        y_labels = self.label_encoder.inverse_transform(y_pred) if self.label_encoder else y_pred

        results = pd.DataFrame({
            "ID": test_df.get("ID", range(len(test_df))),
            "predicted_rainfall": y_labels,
            "confidence": np.max(y_prob, axis=1)
        })
        for i, cls in enumerate(self.label_encoder.classes_):
            results[f"prob_{cls}"] = y_prob[:, i]

        if y_true_enc is not None:
            acc = accuracy_score(y_true_enc, y_pred)
            f1m = f1_score(y_true_enc, y_pred, average="macro")
            print("\n=== External Test Performance ===")
            print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true_enc, y_pred, target_names=self.label_encoder.classes_))
            print("Confusion Matrix:\n", confusion_matrix(y_true_enc, y_pred))

        return results

    # --------- Save Zindi submission (matches SampleSubmission.csv format) ----------
    def save_submission(self, test_file: str, sample_submission_file: str, output_file: str = "submission.csv"):
        if self.final_model is None or self.label_encoder is None:
            raise ValueError("Train the model first before saving submission.")

        test_df = pd.read_csv(test_file)
        X_test = test_df[[c for c in test_df.columns if c not in ["ID", "prediction_time", "user_id", "Target"]]].copy()
        preds = self.final_model.predict(X_test)
        pred_labels = self.label_encoder.inverse_transform(preds)

        sample = pd.read_csv(sample_submission_file).copy()
        # assumes first column is ID, second column is the label to fill
        sample.iloc[:, 1] = pred_labels
        sample.to_csv(output_file, index=False)
        print(f"✅ Submission saved to {output_file}")
        return sample