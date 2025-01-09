# Housing Price Prediction

## Problem Statement
The goal is to predict the median housing prices in California based on features such as location, number of rooms, and proximity to the ocean.

## Dataset Description
Link :  https://www.kaggle.com/datasets/camnugent/california-housing-prices

The **California Housing** dataset contains 20,640 observations with the following variables:
- **longitude**, **latitude**: Geographic coordinates.
- **housing_median_age**: Median age of houses.
- **total_rooms**, **total_bedrooms**: Total number of rooms and bedrooms.
- **population**, **households**: Population and number of households.
- **median_income**: Median income of households.
- **median_house_value**: Median house value (target).
- **ocean_proximity**: Proximity to the ocean (categorical).

## Exploratory Data Analysis (EDA)
- Distribution of housing prices.
- Correlation matrix.
- Relationship between median income and housing prices.

## Modeling
Comparison of several models:
- **Linear Regression**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**

## Evaluation
The best model is selected based on **MSE** (Mean Squared Error).

## Deployment
A Microsoft AzureWebSites is created to predict housing prices in real-time.

## Conclusion
The **XGBoost** model achieved the best performance with an MSE of 2, 238, 153, 643.45
