import pandas as pd
import datetime


def create_features(df):
    current_year = datetime.datetime.now().year

    # Age of property
    df['Age_of_Property'] = current_year - df['Year_Built']

    # Price per sqft (safety)
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']

    # Investment Score (multi-factor)
    df['Investment_Score'] = (
        (df['BHK'] >= 3).astype(int) +
        (df['Nearby_Schools'] > 2).astype(int) +
        (df['Nearby_Hospitals'] > 2).astype(int) +
        (df['Parking_Space'] > 0).astype(int)
    )

    # Good Investment Label
    median_price = df['Price_in_Lakhs'].median()
    df['Good_Investment'] = (df['Price_in_Lakhs'] <= median_price).astype(int)

    # Future Price (Regression Target)
    r = 0.08
    t = 5
    df['Future_Price'] = df['Price_in_Lakhs'] * ((1 + r) ** t)

    return df
