from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TEST_SIZE, RANDOM_STATE, FRAUD_LABEL, DROP_COLUMNS

def preprocess(df):

    # Separate target FIRST
    y = df[FRAUD_LABEL].copy()
    X = df.drop(columns=[FRAUD_LABEL] + DROP_COLUMNS).copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
