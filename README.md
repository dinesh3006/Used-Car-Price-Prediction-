# 🚗 Car Dheko - Used Car Price Prediction

This project is a complete end-to-end data science solution aimed at predicting the price of used cars. It leverages machine learning to provide accurate estimations through an interactive web application built with Streamlit.

The primary goal is to create an accurate and user-friendly tool for both customers and sales representatives to seamlessly estimate used car prices based on various features like make, model, year, fuel type, and mileage.

## 🚀 Live Demo & Preview

![Streamlit App Demo]
<img width="1920" height="867" alt="Screenshot 2025-10-31 201212" src="https://github.com/user-attachments/assets/7bbf48fb-3689-4186-9839-e8d75e6c46f1" />
<br><br>
<img width="1920" height="867" alt="Screenshot 2025-10-31 201403" src="https://github.com/user-attachments/assets/406bc89b-21e3-4d5c-9dc1-69dca24afc46" />
<br><br>
<img width="1920" height="865" alt="Screenshot 2025-10-31 201500" src="https://github.com/user-attachments/assets/aa5addd7-74f7-42eb-855b-4ba0094b9d14" />
<br><br>
<img width="1920" height="859" alt="Screenshot 2025-10-31 201705" src="https://github.com/user-attachments/assets/e1d6d297-e79c-40e3-b157-9a849c8b897d" />
<br><br>
<img width="1920" height="865" alt="Screenshot 2025-10-31 201736" src="https://github.com/user-attachments/assets/c89ea634-0e91-4bd3-905d-0029f06c5385" />

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
