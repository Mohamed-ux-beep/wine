
# Introduction

Wine quality is often determined by various chemical properties. The objective of this project is to use these properties to build a predictive model that can accurately classify the quality of wine. By leveraging machine learning algorithms, we aim to develop a robust model that can assist wine producers in ensuring the quality of their products.

# Dataset 

The dataset used in this project is sourced from the UCI Machine Learning Repository and consists of one file:

 * `WineQT.csv`

Each record in the dataset represents a wine sample, with 11 physicochemical properties and a quality rating.

# Features

The dataset contains the following features:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol
* Quality (target variable)

# Modeling

Random Forest Regressor was used for modeling 

# Evaluation

The models are evaluated using R square to ensure robust performance. Feature importance and residual analysis are also conducted to understand the model behavior better.