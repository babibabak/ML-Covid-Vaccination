# ML-Covid-Vaccination
A machine learning project analyzing global COVID-19 vaccination data to predict total vaccinations using various models, including MLPClassifier. The project includes data preprocessing, exploratory data analysis, model training, and evaluation, implemented in Python using libraries like Pandas, Scikit-learn, and Matplotlib.
# COVID-19 Vaccination Prediction

## Overview
This project leverages machine learning to predict total COVID-19 vaccinations based on vaccination-related features from a global dataset. The dataset includes metrics like people vaccinated, daily vaccinations, and vaccination rates per hundred. The project employs an MLPClassifier for prediction, with data preprocessing, exploratory data analysis (EDA), and model evaluation.

## Features
- **Data Preprocessing**: Handles missing values, feature scaling, and data splitting.
- **Exploratory Data Analysis**: Visualizes relationships between vaccination metrics using scatter plots.
- **Machine Learning**: Implements an MLPClassifier to predict total vaccinations.
- **Evaluation**: Assesses model performance with training and testing accuracy.

## Dataset
The dataset (`country_vaccinations.csv`) contains global COVID-19 vaccination data, including:
- Country, ISO code, date
- Total vaccinations, people vaccinated, daily vaccinations
- Vaccination rates per hundred and per million
- Vaccine types and sources

## Requirements
- Python 3.11
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `pycaret`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/covid-vaccination-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3.Run the Jupyter Notebook:
   ```bash
   jupyter notebook Project1.ipynb
```

## Usage
- Load and preprocess the dataset.
- Perform EDA to understand vaccination trends.
- Train the MLPClassifier model.
- Evaluate the model using accuracy metrics.
- Save the trained model using joblib.

## Results
- Training Accuracy: 62.63%
- Testing Accuracy: 45.07%

## Future Improvements
- Experiment with additional models (e.g., RandomForest, XGBoost).
- Improve feature engineering and handle missing data more effectively.
- Enhance model performance through hyperparameter tuning.
