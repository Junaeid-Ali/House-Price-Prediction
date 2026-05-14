import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =========================
# Load Data
# =========================

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# =========================
# Remove Outliers
# =========================

train_df = train_df.drop(
    train_df[
        (train_df['GrLivArea'] > 4000) &
        (train_df['SalePrice'] < 300000)
    ].index
)

# =========================
# Log Transform
# =========================

train_df['SalePrice_Log'] = np.log1p(train_df['SalePrice'])

# =========================
# Split Features & Target
# =========================

X = train_df.drop(['SalePrice', 'SalePrice_Log'], axis=1)
y = train_df['SalePrice_Log']

# =========================
# Combine Data
# =========================

all_data = pd.concat([X, test_df], axis=0)

# =========================
# Missing Value Handling
# =========================

none_cols = [
    'PoolQC',
    'MiscFeature',
    'Alley',
    'Fence',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'MasVnrType'
]

for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

zero_cols = [
    'GarageYrBlt',
    'GarageArea',
    'GarageCars',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'MasVnrArea'
]

for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

all_data['LotFrontage'] = all_data.groupby(
    'Neighborhood'
)['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

for col in all_data.select_dtypes(include='object'):
    all_data[col] = all_data[col].fillna(
        all_data[col].mode()[0]
    )

for col in all_data.select_dtypes(include=['int64', 'float64']):
    all_data[col] = all_data[col].fillna(
        all_data[col].median()
    )

# =========================
# Feature Engineering
# =========================

all_data['TotalSF'] = (
    all_data['TotalBsmtSF'] +
    all_data['1stFlrSF'] +
    all_data['2ndFlrSF']
)

all_data['TotalBath'] = (
    all_data['FullBath'] +
    (0.5 * all_data['HalfBath']) +
    all_data['BsmtFullBath'] +
    (0.5 * all_data['BsmtHalfBath'])
)

all_data['HouseAge'] = (
    all_data['YrSold'] -
    all_data['YearBuilt']
)

all_data['RemodelAge'] = (
    all_data['YrSold'] -
    all_data['YearRemodAdd']
)

all_data['TotalPorchSF'] = (
    all_data['OpenPorchSF'] +
    all_data['EnclosedPorch'] +
    all_data['3SsnPorch'] +
    all_data['ScreenPorch']
)

all_data['HasGarage'] = (
    all_data['GarageArea'] > 0
).astype(int)

all_data['HasBsmt'] = (
    all_data['TotalBsmtSF'] > 0
).astype(int)

all_data['HasFireplace'] = (
    all_data['Fireplaces'] > 0
).astype(int)

# =========================
# Encoding
# =========================

all_data = pd.get_dummies(all_data)

# =========================
# Split Back
# =========================

X_processed = all_data[:len(X)]

# =========================
# Train Final Model
# =========================

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_processed, y)

# =========================
# Save Model
# =========================

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(X_processed.columns, open('columns.pkl', 'wb'))

print('Model Saved Successfully!')