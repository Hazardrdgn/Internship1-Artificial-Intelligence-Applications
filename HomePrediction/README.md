# Home Price Prediction 🏠

This repository contains a machine learning project developed during my Artificial Intelligence internship. The goal of the project is to analyze real estate data and predict housing prices using various market factors and location-based features.

## 📂 Project Structure

The project relies on multiple datasets to make accurate predictions. The main script and data files are organized as follows:

*   **`HousePriceEstimate.py`**: The core Python script containing the data preprocessing, exploratory data analysis (EDA), and the machine learning model implementation.
*   **`district_prices_monthly.csv`**: Monthly housing price trends categorized by district.
*   **`new_construction.csv`**: Market data specifically focusing on newly constructed properties.
*   **`rentals.csv`**: Data reflecting the rental market trends, which often correlate with property values.
*   **`secondary_sales.csv`**: Sales records for second-hand/pre-owned homes.
*   **`transit_stations.csv`**: Spatial data indicating proximity to public transportation, a key feature in urban real estate valuation.

## 🚀 Objectives
*   Merge and clean diverse real estate datasets (rentals, new constructions, secondary sales).
*   Perform feature engineering (e.g., calculating distance to transit stations).
*   Train a predictive model to estimate property values based on historical and locational data.

## 🛠️ Technologies Used
*   **Python**
*   **Pandas & NumPy** (Data manipulation)
*   **Scikit-Learn** (Machine Learning model)
