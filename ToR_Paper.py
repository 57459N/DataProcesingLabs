# %%
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, auc
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings("ignore")
# %%
root = 'D:/python/DataProcessingLabs'
df = pd.read_csv(f'{root}/data/heart_statlog_cleveland_hungary_final.csv')
# %%
df.head()
# %%
df.info()
# %%
df.describe().T
# %%
df.shape
# %%
df['target'].value_counts(normalize=True)
# %%
num_columns = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
labels = ['Age', 'Resting blood pressure in mm Hg', 'Total cholesterol in mg/dl', 'Maximum heart rate achieved',
          'ST Depression Induced by Exercise']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

axes = axes.flatten()

for i, (column, label) in enumerate(zip(num_columns, labels)):
    sns.histplot(df[column], kde=True, ax=axes[i], bins=20)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Distribution of {label}')

axes[-1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
# %%
nominal_columns = list(set(df.columns) - set(num_columns))
nominal_columns.remove('target')
labels = ['Fasting blood sugar (0=normal, 1=high)', 'Exercise induced angina (0=no, 1=yes)', 'Chest Pain type',
          'Resting electrocardiographic results', 'Sex of patient', 'Slope of peak exercise ST segment']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 16))

axes = axes.flatten()

for i, (column, label) in enumerate(zip(nominal_columns, labels)):
    sns.countplot(data=df, x=column, hue='target', ax=axes[i], palette='Set2')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='Target', loc='upper right')
    axes[i].set_title(f'Distribution of {label}')

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(16, 8))

corr_matrix = df.corr(numeric_only=True)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, cmap='PiYG', annot=True, mask=mask)

plt.title('Correlation Heatmap')

plt.show()
# %%
round(sum(df['cholesterol'] == 0) / len(df) * 100, 1)
# %%
# In real numbers
df[df['cholesterol'] == 0]['target'].value_counts()
# %%
# In proportion
df[df['cholesterol'] == 0]['target'].value_counts(normalize=True)
# %%
df_no_zero_chol = df[df.cholesterol != 0]
df_no_zero_chol['cholesterol'].describe()
# %%
df['cholesterol'].replace(0, 240, inplace=True)
# %%
round(sum(df['resting bp s'] == 0) / len(df) * 100, 3)
# %%
df[df['resting bp s'] == 0]
# %%
df['resting bp s'].describe()
# %%
df['resting bp s'].replace(0, 130, inplace=True)
# %%
round(sum(df['oldpeak'] < 0) / len(df) * 100, 1)
# %%
df['oldpeak'].describe()
# %%
df[df < 0] = 0.6
# %%

# Set a variable X to the features of the dataframe and y to the target column.
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# %%
M_features = X.shape[1]
max_features = math.floor(np.sqrt(M_features))
max_features
# %%

RF = RandomForestClassifier(oob_score=True,
                            random_state=42,
                            warm_start=True,
                            n_jobs=-1,
                            max_features=max_features)

oob_list = list()

for n_trees in [15, 30, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500]:
    RF.set_params(n_estimators=n_trees)
    RF.fit(X_train, y_train)
    oob_error = 1 - RF.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

rf_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')
rf_oob_df
# %%
sns.set_context('talk')
sns.set_style('white')

ax = rf_oob_df.plot(legend=False, marker='o', figsize=(14, 7), linewidth=5)
ax.set(ylabel='out-of-bag error')
plt.savefig(f'{root}/checkpoints/plots/heart_trees_oob')
# %%

clf = RandomForestClassifier()
param_grid = {"n_estimators": 500,
              "max_depth": [3, None],
              "max_features": max_features,
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": True,
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X, y)
# %%
grid_search.best_params_
# %%
rf_clf = RandomForestClassifier(n_estimators=500,
                                criterion='entropy',
                                max_depth=None,
                                min_samples_leaf=1,
                                min_samples_split=3,
                                max_features=max_features,
                                bootstrap=True)
rf_clf.fit(X_train, y_train)
# %% md

# %%

y_pred = rf_clf.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)
# %%

plt.figure(figsize=(20, 20))
sns.set_context('talk')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_clf.classes_)
disp.plot(cmap="Greens_r")
plt.savefig(f'{root}/checkpoints/plots/heart_confusion_matrix')
plt.show()
# %%

plt.figure(figsize=(10, 10))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                       estimator_name='Random Forest')
disp.plot()
plt.savefig(f'{root}/checkpoints/plots/heart_rocauc')
plt.show()
# %%
feature_imp = pd.Series(rf_clf.feature_importances_, index=X.columns).sort_values(ascending=False)

ax = feature_imp.plot(kind='bar', figsize=(16, 6))
ax.set(ylabel='Relative Importance')
ax.set(xlabel='Feature')
plt.show()
