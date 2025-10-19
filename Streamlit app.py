# Car Dekho - Used Car Price Prediction Application
# Professional ML-powered web app with enhanced visualizations
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import base64
from streamlit_extras.stylable_container import stylable_container

@st.cache_data
def get_img_as_base64(file):
    """Convert local image file to base64 string for CSS background"""
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Image not found: {file}")
        return None

@st.cache_resource
def load_resources():
    """Load and cache ML model, encoders, scaler, and dataset for performance"""
    base_path = r"C:\Users\hp\OneDrive\Pictures\guvi certificates\Car Dheko - Used Car Price Prediction\streamlit"
    return {'df': pd.read_csv(f"{base_path}\\cars.csv"), 'model': joblib.load(f"{base_path}\\best_xgb_model.pkl"),
            'scaler': joblib.load(f"{base_path}\\scaler.pkl"), 'label_enc': joblib.load(f"{base_path}\\bodytype_label_encoder.pkl"),
            'onehot_enc': joblib.load(f"{base_path}\\one_hot_encoder.pkl")}

def predict_price(inputs, resources):
    """Transform user inputs and predict car price using XGBoost model"""
    num_feats = np.array([int(inputs[k]) for k in ['seats','km','year','owner','engine','gear']] + [float(inputs['mileage'])]).reshape(1,-1)  # Collect numerical features
    scaled = resources['scaler'].transform(num_feats)  # Standardize numerical values
    body_enc = resources['label_enc'].transform([inputs['bodytype']]).reshape(1,-1)  # Encode body type as number
    cat_feats = np.array([inputs[k] for k in ['fuel','trans','insurance','oem','drive','city','model']]).reshape(1,-1)  # Collect categorical features
    cat_enc = resources['onehot_enc'].transform(cat_feats)  # Convert categories to binary vectors
    final = np.hstack((scaled, body_enc, cat_enc))  # Combine all processed features
    return resources['model'].predict(final)[0]  # Return predicted price

# Configure page settings with wide layout and custom icon
st.set_page_config(layout="wide", page_icon="🚗", page_title="Car Price Prediction", initial_sidebar_state="expanded")

# Load background image and apply as CSS background
img_path = r"C:\Users\hp\OneDrive\Pictures\guvi certificates\Car Dheko - Used Car Price Prediction\streamlit\car_image.jpg"
img_b64 = get_img_as_base64(img_path)
if img_b64:
    st.markdown(f'<style>.stApp{{background-image:url("data:image/png;base64,{img_b64}");background-size:cover;background-position:center;background-attachment:fixed;}}</style>', unsafe_allow_html=True)

# Style sidebar with transparent background
st.markdown('<style>[data-testid="stSidebar"]{{background-color:#60191900;}}</style>', unsafe_allow_html=True)
st.title("🚗 :red[Car Dekho - Used Car Price Prediction]")  # Main title

# Add animated car emojis below title
st.markdown('<div style="text-align:center;padding:20px;"><div style="display:inline-block;animation:slide 3s infinite;">🚗💨 🏎️💨 🚙💨</div></div><style>@keyframes slide{0%,100%{transform:translateX(-50px);opacity:0.6;}50%{transform:translateX(50px);opacity:1;}}</style>', unsafe_allow_html=True)

resources = load_resources()  # Load all ML resources
df = resources['df']  # Extract dataframe for dropdowns

# Sidebar for user inputs
with st.sidebar:
    st.title(":red[Vehicle Features]")  # Sidebar title
    inputs = {  # Dictionary to store all user selections
        'trans': st.selectbox("Transmission", df['transmission'].unique()),
        'oem': st.selectbox("Brand", df['oem'].unique()),
        'km': st.selectbox("KMs Driven", sorted(df['kms'].unique().astype(int))),
        'gear': st.selectbox("Gears", sorted(df['Gearbox'].unique().astype(int))),
        'fuel': st.selectbox("Fuel Type", df['fueltype'].unique()),
        'bodytype': st.selectbox("Body Type", df['bodytype'].unique()),
        'mileage': st.selectbox("Mileage (kmpl)", sorted(df['Mileage'].unique())),
        'drive': st.selectbox("Drive Type", df['DriveType'].unique()),
        'year': st.selectbox("Model Year", sorted(df['modelYear'].unique().astype(int))),
        'seats': st.selectbox("Seats", sorted(df['seats'].unique().astype(int))),
        'owner': st.selectbox("Owners", sorted(df['ownerNo'].unique().astype(int))),
        'engine': st.selectbox("Engine CC", sorted(df['engine_cc'].unique().astype(int))),
        'insurance': st.selectbox("Insurance", df['Insurance'].unique()),
        'city': st.selectbox("City", df['city'].unique()),
        'model': st.selectbox("Model", sorted(df['model'].unique()))
    }
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    col_btn = st.columns([1,2,1])  # Create 3 columns for centering
    with col_btn[1]:  # Use middle column
        with stylable_container(key="btn", css_styles="button{background:linear-gradient(90deg,#0575e6,#021b79);color:white;border-radius:20px;width:100%;padding:10px;font-size:16px;font-weight:bold;}"):
            clicked = st.button("🔍 Estimate Price")  # Styled prediction button

# Display results only after button click
if clicked:
    price = predict_price(inputs, resources)  # Get prediction from model
    st.success(f"### Estimated Price: :blue[₹ {price/100000:,.2f} Lakhs]")  # Show formatted price
    
    col1, col2 = st.columns(2)  # Create 2-column layout for charts
    with col1:  # Left column charts
        fig1 = px.bar(df, x='bodytype', y='price', title='💰 Average Price by Body Type', color='bodytype', color_discrete_sequence=px.colors.qualitative.Set3)  # Bar chart with custom colors
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.histogram(df, x='Mileage', nbins=30, title='⛽ Mileage Distribution', color_discrete_sequence=['#0575e6'])  # Histogram in blue
        st.plotly_chart(fig2, use_container_width=True)
    with col2:  # Right column charts
        fig3 = px.scatter(df, x='kms', y='price', color='fueltype', title='🚗 Distance vs Price Analysis', labels={'kms':'Kilometers Driven','price':'Price (₹)'}, color_discrete_sequence=px.colors.qualitative.Pastel)  # Scatter with labels
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.imshow(df[['kms','engine_cc','price','seats','ownerNo']].corr(), text_auto='.2f', title='📊 Feature Correlation Matrix', color_continuous_scale='RdBu_r')  # Heatmap with red-blue colors
        st.plotly_chart(fig4, use_container_width=True)
