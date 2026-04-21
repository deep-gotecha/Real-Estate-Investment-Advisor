import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def run_eda(df):

    st.title("  Real Estate EDA")

    # =========================================================
    #  BASIC INFO
    # =========================================================
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # =========================================================
    #  DISTRIBUTIONS
    # =========================================================
    st.subheader("Price Distribution")
    plt.figure()
    sns.histplot(df['Price_in_Lakhs'])
    st.pyplot(plt)

    st.subheader("Size Distribution")
    plt.figure()
    sns.histplot(df['Size_in_SqFt'])
    st.pyplot(plt)

    # =========================================================
    #  RELATIONSHIPS
    # =========================================================
    st.subheader("Size vs Price")
    plt.figure()
    sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df)
    st.pyplot(plt)

    # =========================================================
    #  TOP CITIES BY PRICE 
    # =========================================================
    st.subheader("Top 10 Expensive Cities")

    city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10)

    plt.figure()
    city_price.plot(kind='bar')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # =========================================================
    #  PRICE PER SQFT BY STATE
    # =========================================================
    st.subheader("Avg Price per SqFt by State")

    state_price = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).head(10)

    plt.figure()
    state_price.plot(kind='bar')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # =========================================================
    #  BHK DISTRIBUTION
    # =========================================================
    st.subheader("BHK Distribution")

    plt.figure()
    sns.countplot(x="BHK", data=df)
    st.pyplot(plt)

    # =========================================================
    #  FURNISHED VS PRICE
    # =========================================================
    st.subheader("Furnished Status vs Price")

    plt.figure()
    sns.boxplot(x='Furnished_Status', y='Price_in_Lakhs', data=df)
    st.pyplot(plt)

    # =========================================================
    #  TRANSPORT VS PRICE 
    # =========================================================
    st.subheader("Transport Accessibility vs Price")

    plt.figure()
    sns.boxplot(x='Public_Transport_Accessibility', y='Price_in_Lakhs', data=df)
    st.pyplot(plt)

    # =========================================================
    #  INVESTMENT INSIGHT 
    # =========================================================
    st.subheader("Investment Insight")

    median_price = df["Price_in_Lakhs"].median()

    good = df[df["Price_in_Lakhs"] <= median_price]
    bad = df[df["Price_in_Lakhs"] > median_price]

    st.write("Good Investment Properties:", len(good))
    st.write("Bad Investment Properties:", len(bad))

    # =========================================================
    #  CORRELATION HEATMAP 
    # =========================================================
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    st.pyplot(plt)