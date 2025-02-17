# Streamlit-Predictor (I'll most definitely change this name later on😂)

## Overview
When working with datasets as a data scientist on a notebook platform like Jupyter, or any platform in general, properly gaining insights after understanding the business requirements and drafting out an analytical approach involves several steps:

- **Collection and Gathering**
- **Data Understanding** (often supported by a provided data dictionary)
- **Data Wrangling (Cleaning)**
- **Exploratory Data Analysis** (visualizations, pivot tables, correlation matrices, etc.)
- **Data Preparation for Model Training** (transformations, formatting, scaling, etc.)
- **Model Selection & Building**
- **Model Training**
- **Model Evaluation and Optimization**
- **Model Deployment** - which leads back to the first step in an iterative cycle, where feedback and evolving data requirements help improve the model's performance or extract more valuable insights.

Every data scientist knows the hassle of creating new notebooks and rewriting code when dealing with new datasets. But what if we could abstract much of that coding process? What if you could handle everything from data wrangling through model evaluation and optimization via an intuitive UI?

**Streamlit-Predictor** is an attempt to simplify this process using **Streamlit**, providing a user-friendly interface to perform essential data science tasks without extensive manual coding.

## Features
- **Dataset Upload**: Upload CSV files and preview data instantly.
- **Data Cleaning**: Handle missing values, duplicates, and outliers with just a few clicks.
- **Exploratory Data Analysis (EDA)**: Generate visualizations like histograms, scatter plots, boxplots, and correlation matrices.
- **Feature Engineering**: Create new features, encode categorical variables, and apply scaling techniques.
- **Model Building**: Select from popular machine learning models like Linear Regression, Decision Trees, Random Forest, and more.
- **Model Training**: Train models on uploaded datasets with real-time progress tracking.
- **Model Evaluation**: View performance metrics such as accuracy, precision, recall, F1-score, and RMSE.
- **Model Deployment**: Deploy models locally or via APIs for real-time predictions.
- **Interactive Predictions**: Input new data points via the UI to get live predictions.


## Usage
To get started, clone the repository and install all dependencies, beginning with streamlit.
Run the stramlit apllication on your bash with the code `Streamlit run Main.py`


## How It Works
1. **Upload Data**: Upload your dataset in CSV format.
2. **Explore Data**: View data summaries, distributions, and visualizations.
3. **Merge Sheets**:  Merge multiple excel sheets on a single common column
4. **Clean Data**: Apply cleaning techniques such as filling missing values and removing duplicates.
5. **Engineer Features**: Create new features, scale data, and encode categorical variables
6. **Select and Train Model**: Choose a model from the available machine learning algorithms.
7. **Evaluate Model**: View model performance metrics and visualizations (in-sample and out sample)


## Technologies Used
- **Python (and featured libraries)**
- **Streamlit** - for the interactive UI

## Feel Free to Add contributions
1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them.
4. Push to your forked repository.
5. Submit a pull request.

## License
This project is licensed under the MIT License.

---

**Streamlit-Predictor** aims to make data science workflows more accessible, efficient, and enjoyable for professionals and enthusiasts alike. Happy modeling! 

