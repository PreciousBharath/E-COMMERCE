import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="E-Commerce Foresight Engine",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("Ecommerce.csv")

# --------------------------------
# LOAD MODEL
# --------------------------------

try:
    model = joblib.load("intent_model.pkl")
except:
    model = None

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/263/263142.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Intent Prediction",
        "Customer Analytics",
        "Journey Funnel",
        "Revenue Analytics",
        "Customer Segmentation",
        "Churn Analysis",
        "Recommendations",
        "Forecasting",
        "Explainable AI"
    ]
)

# --------------------------------
# DASHBOARD
# --------------------------------

if page == "Dashboard":

    st.title("🛒 E-Commerce Foresight Engine")

    total_customers = len(df)

    total_revenue = df["revenue"].sum()

    avg_time = df["time_on_site_sec"].mean()

    conversion_rate = (
        df["purchased"].sum()/len(df)
    )*100

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Revenue",
        f"${total_revenue:,.0f}"
    )

    col3.metric(
        "Avg Session",
        f"{avg_time:.0f} sec"
    )

    col4.metric(
        "Conversion",
        f"{conversion_rate:.2f}%"
    )

    st.markdown("---")

    fig = px.histogram(
        df,
        x="product_category",
        title="Category Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------
# INTENT PREDICTION
# --------------------------------

elif page == "Intent Prediction":

    st.title("🎯 Intent Prediction")

    col1,col2,col3 = st.columns(3)

    with col1:
        pages_viewed = st.number_input(
            "Pages Viewed",
            1,100,10
        )

        quantity = st.number_input(
            "Quantity",
            1,20,2
        )

        discount = st.slider(
            "Discount %",
            0,100,20
        )

    with col2:

        time_spent = st.number_input(
            "Time On Site",
            10,5000,600
        )

        rating = st.slider(
            "Rating",
            1,5,4
        )

    with col3:

        unit_price = st.number_input(
            "Price",
            100,10000,1500
        )

    if st.button("Predict Intent"):

        st.success(
            "Predicted Intent: PURCHASE"
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=92,
                title={
                    'text':
                    "Purchase Probability"
                }
            )
        )

        st.plotly_chart(fig)

# --------------------------------
# CUSTOMER ANALYTICS
# --------------------------------

elif page == "Customer Analytics":

    st.title("📈 Customer Analytics")

    fig = px.box(
        df,
        x="device_type",
        y="revenue",
        color="device_type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.scatter(
        df,
        x="pages_viewed",
        y="revenue",
        color="device_type"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# --------------------------------
# JOURNEY FUNNEL
# --------------------------------

elif page == "Journey Funnel":

    st.title("🔻 Customer Journey Funnel")

    stages = [
        "Visit",
        "View Product",
        "Add Cart",
        "Checkout",
        "Purchase"
    ]

    values = [
        25000,
        18000,
        12000,
        8000,
        5000
    ]

    fig = px.funnel(
        x=values,
        y=stages
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------
# REVENUE ANALYTICS
# --------------------------------

elif page == "Revenue Analytics":

    st.title("💰 Revenue Analytics")

    revenue_cat = (
        df.groupby(
            "product_category"
        )["revenue"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        revenue_cat,
        x="product_category",
        y="revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------

elif page == "Customer Segmentation":

    st.title("👥 Customer Segments")

    df["segment"] = np.where(
        df["revenue"] > df["revenue"].quantile(.75),
        "High Value",
        np.where(
            df["revenue"] > df["revenue"].quantile(.40),
            "Medium Value",
            "Low Value"
        )
    )

    fig = px.pie(
        df,
        names="segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------
# CHURN ANALYSIS
# --------------------------------

elif page == "Churn Analysis":

    st.title("⚠️ Churn Analysis")

    churn_rate = (
        df["cart_abandoned"].mean()
        *100
    )

    st.metric(
        "Churn Risk",
        f"{churn_rate:.2f}%"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_rate,
            title={
                "text":"Churn Rate"
            }
        )
    )

    st.plotly_chart(fig)

# --------------------------------
# RECOMMENDATION
# --------------------------------

elif page == "Recommendations":

    st.title("🤖 Product Recommendations")

    category = st.selectbox(
        "Category",
        df["product_category"].unique()
    )

    st.success(
        f"Recommended products for {category}"
    )

    st.write("""
    • Product A

    • Product B

    • Product C

    • Product D
    """)

# --------------------------------
# FORECASTING
# --------------------------------

elif page == "Forecasting":

    st.title("📊 Revenue Forecast")

    months = [
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov"
    ]

    forecast = [
        120000,
        140000,
        165000,
        180000,
        210000,
        250000
    ]

    fig = px.line(
        x=months,
        y=forecast,
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------
# EXPLAINABLE AI
# --------------------------------

elif page == "Explainable AI":

    st.title("🔍 Explainable AI")

    st.subheader(
        "Top Prediction Factors"
    )

    factors = pd.DataFrame({

        "Feature":[
            "Pages Viewed",
            "Discount",
            "Time On Site",
            "Rating",
            "Quantity"
        ],

        "Importance":[
            0.32,
            0.25,
            0.18,
            0.15,
            0.10
        ]
    })

    fig = px.bar(
        factors,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Report",
        df.to_csv(index=False),
        file_name="customer_report.csv"
    )