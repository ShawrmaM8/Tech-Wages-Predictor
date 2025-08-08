import pandas as pd
import numpy as np

ds = pd.read_csv(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\wage-pred\datasets\DataScience_salaries_2025.csv")
df = pd.DataFrame(ds)
top5_rows = df.head()
info = df.info()

#print(info, '\n')

## Transformation

### Remove outliers in salary_in_usd
q1_usd = df['salary_in_usd'].quantile(0.25)
q3_usd = df['salary_in_usd'].quantile(0.75)
iqr_usd = q3_usd - q1_usd
print("IQR of salary in usd: ", iqr_usd)
ll = q1_usd -1.5*(iqr_usd)
ul = q3_usd +1.5*(iqr_usd)
print('Lower limit of salary_in_usd: ', ll, ' Upper limit of salary_in_usd: ', ul, '\n') # Output: -31375 and 335625 respectively
outliers = (df['salary_in_usd'] < ll) | (df['salary_in_usd'] > ul)
print('Count of outliers: ', outliers.sum())
df = df[(df['salary_in_usd'] >= ll) & (df['salary_in_usd'] <= ul)]
print(df['salary_in_usd']) ### REMOVED

### Abstract all generally useless features (either repeated/unnecessary/extra) that won't assist our goal of predicting wages
df.drop(['work_year', 'salary', 'salary_currency', 'remote_ratio'], axis=1, inplace=True)
### Already done
print(df.info())

### Convert categorical variables to lower case
categorical_features = df.drop('salary_in_usd', axis=1)
num_features = df[['salary_in_usd']]

# `.apply` works on entire rows or columns
# `.applymap` works element-wise to a data frame
cat_features = categorical_features.applymap(lambda x: x.lower())

### Standardize all domain-specific coded categories
#### Check missing vales first
print('Missing values of experience level: ', df['experience_level'].isna().sum()) # None
#### Standardize `experience_level`
df['standardized_experience_level'] = df['experience_level'].map({
  'EN': 'Entry-level',
  'MI': 'Mid-level',
  'SE': 'Senior-level',
  'EX': 'Executive-level'
})
print('\n', df.head(3), '\n')
unique_etype = df['employment_type'].unique()
unique_eres = df['employee_residence'].unique()
unique_cloc = df['company_location'].unique()
unique_csize = df['company_size'].unique()
print(unique_etype, '\n \n', unique_eres, '\n \n', unique_cloc, '\n \n', unique_csize)

#### Standardize `employment_type` & company_size
df['employment_type_std'] = df['employment_type'].map({
  'FT': 'Full-time',
  'PT': 'Part-time',
  'CT': 'Contracted',
  'FL': 'Freelance'
})
df['company_size_stf'] = df['company_size'].map({
  'S': 'Small',
  'M': 'Medium',
  'L': 'Large'
})

print('total missing vals: ', df.isna().sum()) # No values to fill


## Feature Engineering & Preprocessing
cat_features = cat_features
num_features = num_features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', 'passthrough', num_features)
  ]
)

## Features
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = cat_features
y = num_features

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=2)


## Model
from sklearn.linear_model import Ridge
ridge = Ridge()
### we will go with RidgeRegression because:
##- L2 regularization to prevent overfitting by penalizing large coefficients
##- Output is a weighted sum of input features
### but we need to be wary of :
##- Poor performance when data has complex patterns
##- Inability to capture nonlinear relationships 

### Determine optimal parameters
from sklearn.model_selection import GridSearchCV
# Exhaustively tests all specified parameter combinations to determine the parameters that lead to most optimal model performance

params_grid = {
  'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
  'ridge__solver': ['auto', 'svd', 'saga'],
  'ridge__max_iter': [500, 1000, 2000, 3000, 5000]
}

### Scoring will use Mean Squared Error because:
##- no outliers
##- It is also more sensitive to model improvements
##- With better gradient behaviour (smooth and differentiable everywhere)

### Pipeline to scale each feature and apply Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

pipeline = Pipeline(
  steps=[
    ('scalar', StandardScaler()),
    ('ridge', Ridge())
  ]
)

model = GridSearchCV(
  pipeline,
  params_grid,
  cv=5,
  scoring='neg_mean_squared_error',
  n_jobs=-1 # Parallelize computation
)

model.fit(X_tr, y_tr)

print('\n Best parameters: ', model.best_params_)
print('Best MSE score: ', -model.best_score_, '\n')

### Finalize our Model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

### Full Cross-Validation Results
#results = pd.DataFrame(model.cv_results_)
#print('\n', results[['param_ridge__alpha', 'param_ridge__solver', 'param_ridge__max_iter', 'mean_test_score']].sort_values('mean_test_score', ascending=False))


### Save Model
import joblib
joblib.dump(pipeline, 'ridge_opt_pipeline.joblib')


