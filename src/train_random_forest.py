from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE

def train_rf(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    return model
