import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from src.pattern.correlation import CORR3  # Make sure this matches your file name

# Step 1: Initialize and load data
# ccys = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR',]
ccys = ['AUD', 'JPY', 'USD', 'GBP', 'CAD', 'CHF', 'EUR', 'XAU', 'XAG', 'OIL', 'GAS']
a = CORR3(ccys)
a.get_feature()
a.get_future()
a.apply_cross_sectional()


#Step2: flatten, train test split
idx = 20
mode = 2
selected_interval = f'_{idx}d'
target ='logreturn' + selected_interval
X = a.feature(selected_interval, mode=mode).copy()
target_cols = [col for col in a.future.columns if col.endswith(target)]
target_df = a.future[target_cols].copy()
Xy = pd.concat([X, target_df], axis=1)


X_melted = Xy.reset_index().melt(id_vars='index', var_name='col', value_name='value')
X_melted['pair'] = X_melted['col'].apply(lambda x: x.split('_')[0])
X_melted['feature'] = X_melted['col'].apply(lambda x: '_'.join(x.split('_')[1:]))
X_flat = X_melted.pivot_table(index=['index', 'pair'], columns='feature', values='value').reset_index()
X_flat = X_flat.rename(columns={'index': 'date'})
X_flat.dropna(inplace=True)

X_train = X_flat[X_flat.date < '2019-01-01'].drop(columns=['date', 'pair', target])
y_train = X_flat[X_flat.date < '2019-01-01'][target]

X_test= X_flat[X_flat.date >= '2019-01-01'].drop(columns=['date', 'pair', target])
y_test= X_flat[X_flat.date >= '2019-01-01'][target]


# Step 3: XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 4: Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'idx: {idx}, MSE: {mse:.6f}')

# Step 6: Feature Importance
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))

# Step 5: Plot
plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test, label='Actual', alpha=0.5)
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.5)
plt.title(f'XGBoost Prediction of {target} (MSE: {mse:.6f})')
plt.legend()
plt.tight_layout()
plt.show()

