# Car Dekho - Used Car Price Prediction Application
# Professional ML-powered web app with enhanced visualizations
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import base64
import datetime
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
    try:
        return {'df': pd.read_csv(f"{base_path}\\cars.csv"), 'model': joblib.load(f"{base_path}\\best_xgb_model.pkl"),
                'scaler': joblib.load(f"{base_path}\\scaler.pkl"), 'label_enc': joblib.load(f"{base_path}\\bodytype_label_encoder.pkl"),
                'onehot_enc': joblib.load(f"{base_path}\\one_hot_encoder.pkl")}
    except FileNotFoundError:
        st.error("Error: Model or data files not found. Please check the 'base_path' variable in the code.")
        return None

#
# ==============================================================================
# ### THIS IS THE UPDATED & COMPACT FUNCTION ###
# ==============================================================================
#
def predict_price(inputs, resources):
    """Transform user inputs and predict car price using XGBoost model"""
    # 1. Calculate the 'modelYr' (Age) just like you did in your notebook
    age = datetime.datetime.now().year - int(inputs['year']) 

    # 2. Build the numerical features array
    num_feats = np.array([
        int(inputs['seats']), int(inputs['km']), age, int(inputs['owner']),
        int(inputs['engine']), int(inputs['gear']), float(inputs['mileage'])
    ]).reshape(1, -1)
    
    # 3. Transform features using the loaded encoders
    scaled = resources['scaler'].transform(num_feats)
    body_enc = resources['label_enc'].transform([inputs['bodytype']]).reshape(1,-1)
    
    # 4. Build categorical features array
    cat_feats = np.array([
        inputs['fuel'], inputs['trans'], inputs['insurance'], inputs['oem'],
        inputs['drive'], inputs['city'], inputs['model']
    ]).reshape(1, -1)
    
    cat_enc = resources['onehot_enc'].transform(cat_feats)
    
    # 5. Combine all features and predict
    final = np.hstack((scaled, body_enc, cat_enc))
    return resources['model'].predict(final)[0]
# ==============================================================================
# ### END OF UPDATED FUNCTION ###
# ==============================================================================
#

# Configure page settings
st.set_page_config(layout="wide", page_icon="🚗", page_title="Car Price Prediction", initial_sidebar_state="expanded")

# Load background image
img_path = r"C:\Users\hp\OneDrive\Pictures\guvi certificates\Car Dheko - Used Car Price Prediction\streamlit\car_image.jpg"
img_b64 = get_img_as_base64(img_path)
if img_b64:
    st.markdown(f'<style>.stApp{{background-image:url("data:image/png;base64,{img_b64}");background-size:cover;background-position:center;background-attachment:fixed;}}</style>', unsafe_allow_html=True)

# Style sidebar
st.markdown('<style>[data-testid="stSidebar"]{{background-color:#60191900;}}</style>', unsafe_allow_html=True)
st.title("🚗 :red[Car Dekho - Used Car Price Prediction]")

# Animated car emojis
st.markdown('<div style="text-align:center;padding:20px;"><div style="display:inline-block;animation:slide 3s infinite;">🚗💨 🏎️💨 🚙💨</div></div><style>@keyframes slide{0%,100%{transform:translateX(-50px);opacity:0.6;}50%{transform:translateX(50px);opacity:1;}}</style>', unsafe_allow_html=True)

resources = load_resources()  # Load all ML resources

# Check if resources loaded successfully
if resources:
    df = resources['df']
    valid_selection = True
    
    # Sidebar for user inputs
    with st.sidebar:
        st.title(":red[Vehicle Features]")
        inputs = {}
        
        # 1. Brand (OEM)
        inputs['oem'] = st.selectbox("Brand", sorted(df['oem'].unique()))

        # 2. Model - Filtered by Brand
        df_brand_filtered = df[df['oem'] == inputs['oem']]
        model_options = sorted(df_brand_filtered['model'].unique())
        
        if not model_options:
            st.warning(f"No models found in data for brand: {inputs['oem']}")
            valid_selection = False
            inputs['model'] = None
            df_model_specific = pd.DataFrame(columns=df.columns) 
        else:
            inputs['model'] = st.selectbox("Model", model_options)
            df_model_specific = df_brand_filtered[df_brand_filtered['model'] == inputs['model']]

        # 3. Year - Filtered by Model, extended to current year
        if valid_selection and not df_model_specific.empty:
            available_years_in_data = sorted(df_model_specific['modelYear'].unique().astype(int))
            
            if not available_years_in_data:
                st.error(f"Data for {inputs['oem']} {inputs['model']} is missing 'Model Year'.")
                valid_selection = False
                inputs['year'] = None
            else:
                min_year_in_data = min(available_years_in_data)
                current_year = datetime.datetime.now().year
                year_options = sorted(list(range(min_year_in_data, current_year + 1)), reverse=True)
                inputs['year'] = st.selectbox("Model Year", year_options)
                st.info(f"Predicting for years: {min(year_options)} - {max(year_options)}")
        else:
            inputs['year'] = st.selectbox("Model Year", [], disabled=True)
            if valid_selection: st.error(f"No model selected or available for {inputs['oem']}")
            valid_selection = False
        
        st.markdown("---")
        st.subheader(":red[Vehicle Specifications]")

        # --- Other features filtered by model ---
        if not df_model_specific.empty:
            inputs['trans'] = st.selectbox("Transmission", sorted(df_model_specific['transmission'].unique()))
            inputs['fuel'] = st.selectbox("Fuel Type", sorted(df_model_specific['fueltype'].unique()))
            inputs['bodytype'] = st.selectbox("Body Type", sorted(df_model_specific['bodytype'].unique()))
            inputs['mileage'] = st.selectbox("Mileage (kmpl)", sorted(df_model_specific['Mileage'].unique()))
            inputs['drive'] = st.selectbox("Drive Type", sorted(df_model_specific['DriveType'].unique()))
            inputs['seats'] = st.selectbox("Seats", sorted(df_model_specific['seats'].unique().astype(int)))
            inputs['gear'] = st.selectbox("Gears", sorted(df_model_specific['Gearbox'].unique().astype(int)))
            inputs['engine'] = st.selectbox("Engine CC", sorted(df_model_specific['engine_cc'].unique().astype(int)))
        else:
            # Show disabled dropdowns
            st.selectbox("Transmission", [], disabled=True); st.selectbox("Fuel Type", [], disabled=True)
            st.selectbox("Body Type", [], disabled=True); st.selectbox("Mileage (kmpl)", [], disabled=True)
            st.selectbox("Drive Type", [], disabled=True); st.selectbox("Seats", [], disabled=True)
            st.selectbox("Gears", [], disabled=True); st.selectbox("Engine CC", [], disabled=True)
            valid_selection = False

        st.markdown("---")
        st.subheader(":red[Listing Details]")
        
        # --- Listing features (not model-specific) ---
        inputs['km'] = st.selectbox("KMs Driven", sorted(df['kms'].unique().astype(int)))
        inputs['owner'] = st.selectbox("Owners", sorted(df['ownerNo'].unique().astype(int)))
        inputs['insurance'] = st.selectbox("Insurance", sorted(df['Insurance'].unique()))
        inputs['city'] = st.selectbox("City", sorted(df['city'].unique()))
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn = st.columns([1,2,1])
        with col_btn[1]:
            with stylable_container(key="btn", css_styles="button{background:linear-gradient(90deg,#0575e6,#021b79);color:white;border-radius:20px;width:100%;padding:10px;font-size:16px;font-weight:bold;}"):
                clicked = st.button("🔍 Estimate Price", disabled=(not valid_selection))

    # --- Main Page Display (COMPACTED) ---
    if clicked and valid_selection:
        try:
            price = predict_price(inputs, resources)
            st.success(f"### Estimated Price: :blue[₹ {price/100000:,.2f} Lakhs]")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.bar(df, x='bodytype', y='price', title='💰 Average Price by Body Type', color='bodytype', color_discrete_sequence=px.colors.qualitative.Set3), use_container_width=True)
                st.plotly_chart(px.histogram(df, x='Mileage', nbins=30, title='⛽ Mileage Distribution', color_discrete_sequence=['#0575e6']), use_container_width=True)
            with col2:
                st.plotly_chart(px.scatter(df, x='kms', y='price', color='fueltype', title='🚗 Distance vs Price Analysis', labels={'kms':'Kilometers Driven','price':'Price (₹)'}, color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
                st.plotly_chart(px.imshow(df[['kms','engine_cc','price','seats','ownerNo']].corr(), text_auto='.2f', title='📊 Feature Correlation Matrix', color_continuous_scale='RdBu_r'), use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("This might be due to an unusual combination of features. Please try different options.")

    elif clicked and not valid_selection:
        st.error("Cannot estimate price. Please correct the invalid selections in the sidebar.")
else:
    st.error("Application could not start. Failed to load necessary model or data files.")
    st.info("Please ensure the file paths in the `load_resources` function and `img_path` variable are correct.")
