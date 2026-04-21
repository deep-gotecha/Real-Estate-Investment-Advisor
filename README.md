# Real Estate Investment Advisor

**Predicting Property Profitability & Future Value using Machine Learning**

---

## Overview

This project is an **end-to-end Machine Learning application** designed to assist users in making **data-driven real estate investment decisions**.

It combines:

- **Classification** → Determines whether a property is a _Good Investment_
- **Regression** → Predicts the _future price (5-year forecast)_

The system is deployed using **Streamlit**, providing an interactive dashboard for predictions and data insights.

---

## Problem Statement

Real estate investment decisions are often:

- Based on intuition rather than data
- Lacking future price estimation
- Not supported by structured analysis

### Objective

To build a system that:

- Classifies properties as **Good / Bad Investment**
- Predicts **future property price**
- Provides **analytical insights using EDA**
- Offers a **user-friendly interface**

---

## Features

- Investment prediction (Good / Not Good)
- Future price estimation (5 years)
- Advanced Exploratory Data Analysis (EDA)
- ML Pipeline (Preprocessing + Model)
- Interactive Streamlit UI

---

## Dataset Description

The dataset contains real estate information such as:

| Feature                        | Description             |
| ------------------------------ | ----------------------- |
| State, City                    | Location details        |
| Property_Type                  | Apartment, Villa, House |
| BHK                            | Number of rooms         |
| Size_in_SqFt                   | Area of property        |
| Price_in_Lakhs                 | Property price          |
| Year_Built                     | Construction year       |
| Furnished_Status               | Furnishing level        |
| Nearby_Schools, Hospitals      | Infrastructure          |
| Public_Transport_Accessibility | Connectivity            |
| Parking, Security, Amenities   | Facilities              |

---

## Project Workflow

### 1. Data Preprocessing

- Removed duplicates
- Handled missing values
- Converted categorical variables

---

### 2. Feature Engineering

- Age of Property
- Price per SqFt
- Investment Label (based on median price)
- Future Price using compound growth

---

### 3. Model Building

Two models were trained:

- **Classification Model**
  - Algorithm: Random Forest Classifier
  - Output: Good Investment / Not Good

- **Regression Model**
  - Algorithm: Random Forest Regressor
  - Output: Future Price

---

### 4. Pipeline

Used:

- `ColumnTransformer`
- `StandardScaler`
- `OneHotEncoder`

Ensures consistent preprocessing and avoids manual errors

---

### 5. Deployment

Built using **Streamlit**:

- Input property details
- Get predictions instantly
- View data insights

---

## Exploratory Data Analysis (EDA)

The application includes an advanced EDA dashboard:

- Price & Size Distribution
- Top Cities by Price
- State-wise Price per SqFt
- BHK Distribution
- Furnished vs Price
- Transport vs Price
- Correlation Heatmap

Helps users understand **market trends and patterns**

---

## Project Structure

```
real_estate_project/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
├── src/
│   ├── eda.py                 # EDA functions
│   ├── preprocessing.py       # Data cleaning
│   ├── feature_engineering.py # Feature creation
│   └── predict.py             # Prediction logic
│
├── models/
│   ├── classifier_pipeline.pkl
│   ├── regressor_pipeline.pkl
│   └── feature_columns.pkl
│
├── data/
├── train.py                   # Model training script
├── requirements.txt
└── README.md
```

---

## ▶ How to Run

```bash
# Clone repository
git clone https://github.com/deep-gotecha/real-estate-investment-advisor.git

# Navigate to project
cd real-estate-investment-advisor

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/streamlit_app.py
```

---

## Output

- Investment Decision (Good / Not Good)
- Predicted Future Price
- Data Insights Dashboard

---

## Note

- Models are pre-trained and stored in `/models`

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn

---

## Future Improvements

- Feature importance visualization
- SHAP-based explainability
- City-level filtering
- Cloud deployment (Streamlit Cloud / AWS)

---

## Conclusion

This project demonstrates a **complete machine learning lifecycle**:

> Data → Preprocessing → Modeling → Deployment → Insights

It transforms raw real estate data into a **decision-support system**, enabling smarter and more informed investment choices.

---

## Author

**Deep Gotecha**
