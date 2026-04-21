import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df.fillna({
        'Parking_Space': 0,
        'Nearby_Schools': df['Nearby_Schools'].median(),
        'Nearby_Hospitals': df['Nearby_Hospitals'].median()
    }, inplace=True)

    # Convert categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
