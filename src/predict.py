import joblib
import numpy as np


def predict(input_data, clf, reg, scaler):
    input_scaled = scaler.transform([input_data])

    investment = clf.predict(input_scaled)[0]
    future_price = reg.predict(input_scaled)[0]

    return investment, future_price
