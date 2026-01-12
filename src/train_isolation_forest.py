from sklearn.ensemble import IsolationForest

def train_if(X_train,y_train):
    model = IsolationForest(
        n_estimators=300,
        contamination=0.015,
        random_state=42
    )
    model.fit(X_train,y_train)
    return model