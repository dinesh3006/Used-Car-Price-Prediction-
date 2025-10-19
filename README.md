# 🚗 Car Dheko - Used Car Price Prediction

This project is a complete end-to-end data science solution aimed at predicting the price of used cars. It leverages machine learning to provide accurate estimations through an interactive web application built with Streamlit.

The primary goal is to create an accurate and user-friendly tool for both customers and sales representatives to seamlessly estimate used car prices based on various features like make, model, year, fuel type, and mileage.

## 🚀 Live Demo & Preview

![Streamlit App Demo]
<img width="1920" height="869" alt="Screenshot 2025-10-20 040121" src="https://github.com/user-attachments/assets/4b0d1182-5dad-4e40-804e-49a804328e3a" />
<br><br>
<img width="1920" height="858" alt="Screenshot 2025-10-20 040731" src="https://github.com/user-attachments/assets/85050b0b-af50-4da9-9e46-4d9dee42f6e9" />
<br><br>
<img width="1920" height="857" alt="Screenshot 2025-10-20 040619" src="https://github.com/user-attachments/assets/65aaff14-8381-4791-a066-463c88be17fa" />

---

## 🛠 Tech Stack

This project utilizes a modern data science and web development stack:

-   **Data Analysis:** Pandas, NumPy
-   **Data Visualization:** Plotly Express, Matplotlib, Seaborn
-   **Machine Learning:** Scikit-learn (for preprocessing, scaling, and encoding), XGBoost (for the regression model)
-   **Model Deployment:** Streamlit
-   **Core Language:** Python 3.x
-   **Notebooks:** Jupyter Notebook

---

## 📈 Project Workflow

The project follows a structured data science pipeline:

1.  **Data Collection:** Initial data was gathered and consolidated into a single structured dataset (`structure_cars_final.csv`).
2.  **Data Preprocessing:** The raw data was extensively cleaned, handling missing values, correcting data types, and removing outliers using the IQR method. The cleaned dataset (`cars_eda.csv`) was saved for modeling. (See: `datapreprocessing.ipynb`)
3.  **Exploratory Data Analysis (EDA):** In-depth analysis was performed to understand the relationships between different car features (e.g., `kms_driven`, `fuel_type`, `body_type`) and their impact on price.
4.  **Feature Engineering:** Categorical features were transformed using One-Hot Encoding and Label Encoding. Numerical features were standardized using `StandardScaler` to prepare them for the model.
5.  **Model Training & Selection:** Several regression models were trained and evaluated. The **XGBoost Regressor** was selected as the best-performing model based on metrics like R-squared and RMSE.
6.  **Model Persistence:** The trained model (`best_xgb_model.pkl`), scaler, and encoders were saved as pickle files (`.pkl`) for use in the application. (See: `modelling.ipynb`)
7.  **Web App Deployment:** A user-friendly web interface was built using **Streamlit**. This app loads the saved models to make real-time price predictions based on user input. (See: `app.py`)

---

## 📂 File Structure

A professional repository layout is recommended. I suggest organizing your files as follows:
