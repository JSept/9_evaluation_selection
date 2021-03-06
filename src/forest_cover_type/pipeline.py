from msilib.schema import Feature
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def create_pipeline(
    regression: str, use_scaler: bool, max_iter: int, logreg_C: float, random_state: int, feature_selection: bool
) -> Pipeline:
    pipeline_steps = []
    if feature_selection:
        rf = RandomForestClassifier(n_estimators=500, random_state=random_state)
        pipeline_steps.append(('fsl', SelectFromModel(estimator=rf)))
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if regression == "logreg":
        classifier = LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            )
    if regression == "logregcv":
        classifier = LogisticRegressionCV(
                random_state=random_state, max_iter=max_iter, cv=5
            )
    pipeline_steps.append(("classifier", classifier))
    return Pipeline(steps=pipeline_steps)