# %% [markdown]
# ## StumbleUpon Evergreen Classification Challenge
# [Kaggle Competition](https://www.kaggle.com/c/stumbleupon/overview)

# %% [markdown]
# ## Import Libraries:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json

# %% [markdown]
# ## Load Data:

# %%
data_train = pd.read_csv('train.tsv', sep='\t')
data_test = pd.read_csv('test.tsv', sep='\t')

# %% [markdown]
# ## Data Exploration and Cleaning:

# %%
# Extract boilerplate features
def extract_title_body_length(data):
    boilerplatedf = data["boilerplate"].apply(json.loads)
    boilerplatedf = pd.DataFrame(boilerplatedf.tolist())
    data['boilerplate_title'] = boilerplatedf['title'].fillna('')
    data['boilerplate_body'] = boilerplatedf['body'].fillna('')
    data['boilerplate_title_length'] = data['boilerplate_title'].apply(len)
    data['boilerplate_body_length'] = data['boilerplate_body'].apply(len)
    return data

data_train = extract_title_body_length(data_train)
data_test = extract_title_body_length(data_test)

# Clean "is_news" and "news_front_page"
data_train['is_news'] = data_train['is_news'].str.replace("?", "0").astype(int)
data_test['is_news'] = data_test['is_news'].str.replace("?", "0").astype(int)

data_train['news_front_page'] = data_train['news_front_page'].str.replace("?", "0").astype(int)
data_test['news_front_page'] = data_test['news_front_page'].str.replace("?", "0").astype(int)

# Drop unnecessary columns
droped_columns = ['boilerplate', 'url', 'boilerplate_body', 'boilerplate_title', 'urlid', 'alchemy_category_score']
data_test_ft = data_test.copy()
data_train.drop(columns=droped_columns, inplace=True)
data_test.drop(columns=droped_columns, inplace=True)

# %% [markdown]
# ## Encoding Categorical Variables with OneHotEncoder:

# Identify categorical and numeric columns
categorical_columns = data_train.select_dtypes(include=["object"]).columns
numeric_columns = data_train.select_dtypes(include=["number"]).columns.drop('label')

# Apply OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # Keep numeric columns as is
)

# Fit and transform the training data
X_train_full = data_train.drop(columns=['label'])
y_train_full = data_train['label']

X_train_preprocessed = preprocessor.fit_transform(X_train_full)

# Apply the same transformation to the test data
X_test_full = data_test
X_test_preprocessed = preprocessor.transform(X_test_full)

# %% [markdown]
# ## Scaling Numerical Features:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_preprocessed)
X_test_scaled = scaler.transform(X_test_preprocessed)

# %% [markdown]
# ## Modeling:

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_full, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Evaluate model on validation set
y_val_pred = model.predict(X_val)
print("Validation Performance:\n", classification_report(y_val, y_val_pred))

# %% [markdown]
# ## Generate Predictions for Test Set and Submission File:

# Predict on test data
y_test_pred = model.predict(X_test_scaled)

# Prepare submission file
submission = pd.DataFrame({
    'urlid': data_test_ft['urlid'],  # Assuming test.csv contains a 'urlid' column
    'label': y_test_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
