# Data Science Challenge - Predicting User Activity

## Project Overview

This project aims to predict the number of days a user will be active in the first 28 days after re-registering for a game. By analyzing the user’s prior activity, session behavior, and other attributes, we can better understand engagement patterns and anticipate user activity, which can help improve personalized gaming experiences.

The solution includes a data processing pipeline, feature engineering, model training and tuning, model explainability using SHAP (SHapley Additive exPlanations), and a submission-ready prediction file.

## Setup and Requirements

To run this project, you will need:
	•	Python 3.8+
	•	Required Python libraries: pip install pandas numpy matplotlib seaborn scikit-learn shap joblib

 ## Data Preprocessing

1. Load Data: We load the training and test datasets for both user registration and previous activity data, merging them on the unique user_id to create a unified dataset for training and testing.
2. Data Cleaning: Missing values are handled by:
  	•	Filling missing categorical values with the mode.
  	•	Filling missing numerical values with the median.
3. Encoding Categorical Features: Label encoding is applied to categorical features in the training and test sets, while boolean values are converted to integer format for compatibility with the model pipeline.

##  Feature Engineering

The feature engineering phase incorporates domain knowledge to create meaningful features that can improve the model’s predictive performance. Key engineered features include:
1. Interaction Terms:
	  •	playtime_avg_stars_interaction: Interaction between playtime and avg_stars_top_11_players.
2. Categorical Binning:
	  •	playtime_category: Categorizes playtime into ‘low’, ‘medium’, ‘high’, and ‘very high’ bins.
3. High-Engagement Indicator:
    •	high_engagement: Indicates users with a high session_count and substantial playtime, likely representing more engaged players.

## Model Training and Evaluation

1. Pipeline Setup:
	  •	We use Pipeline from sklearn.pipeline to automate preprocessing (scaling and encoding) and model training.
	  •	ColumnTransformer is used to handle both numerical scaling and categorical encoding.
2. Model Choice:
	  •	GradientBoostingRegressor was selected for its strong performance with structured data.
3. Cross-Validation:
	  •	10-fold cross-validation was conducted using neg_mean_absolute_error as the scoring metric, and the mean and standard deviation of the MAE are displayed to evaluate the model’s consistency.
4. Evaluation Visualization:
	  •	A box plot of cross-validation MAE scores provides insights into model performance variability.

## Model Explainability

We employ SHAP (SHapley Additive exPlanations) for model interpretability, allowing us to understand feature contributions:
1. Summary Plot:
	  •	shap.summary_plot shows global feature importance, helping us understand which features most influence predictions.
2. Preparation for SHAP:
	  •	The Pipeline transformation changes the feature structure, so the transformed dataset (X_transformed) is used for SHAP calculations, along with adjusted feature names for clarity.

## Saving and Downloading the Model

The trained model pipeline, including preprocessing steps, is saved as output_model.pkl.pkl using joblib, making it reusable without retraining.                        
