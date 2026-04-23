import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── LOAD DATASET ────────────────────────────────────────────────────────────
# Dataset: healthcare-dataset-stroke-data.csv from Kaggle
# Place the CSV file in the data/ folder before running this script.
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

print(df.head())
print(df.value_counts())

# ─── NULL VALUE ANALYSIS ─────────────────────────────────────────────────────
print(df.isna().sum())
df.isna().sum().plot.barh()
plt.title("Missing values per column")
plt.show()

print(df.describe())
print(df.info())

# ─── PRE-PROCESSING + EDA ────────────────────────────────────────────────────

df = df.drop(['id'], axis=1)

# Gender analysis
print(df['gender'].value_counts())
df['gender'] = df['gender'].replace('Other', 'Female')
df['gender'].value_counts().plot(kind="pie", title="Gender Distribution")
plt.show()

# Target feature — Stroke
print(df['stroke'].value_counts())
df['stroke'].value_counts().plot(kind="bar", color="goldenrod", title="Stroke Count")
plt.show()
print("% of people who actually got a stroke:",
      (df['stroke'].value_counts()[1] / df['stroke'].value_counts().sum()).round(3) * 100)

# Hypertension Analysis
df['hypertension'].value_counts().plot(kind="bar", color="indianred", title="Hypertension")
plt.show()

# Work type Analysis
print(df['work_type'].value_counts())
df['work_type'].value_counts().plot(kind="pie", title="Work Type Distribution")
plt.show()

# Smoking status Analysis
print(df['smoking_status'].value_counts())
df['smoking_status'].value_counts().plot.bar(stacked=True, title="Smoking Status")
plt.show()
df['smoking_status'].value_counts().plot.pie(
    labels=["Never Smoked", "Unknown", "Formerly Smoked", "Smokes"],
    colors=["mediumseagreen", "khaki", "lightskyblue", "lightcoral"],
    explode=[0.1, 0, 0, 0],
    autopct="%.0f%%",
    figsize=(4, 4)
)
plt.show()

# Residence type Analysis
print(df['Residence_type'].value_counts())
df['Residence_type'].value_counts().plot(kind="bar", title="Residence Type")
plt.show()

# BMI analysis
print("BMI null count:", df['bmi'].isnull().sum())
sns.histplot(data=df['bmi'])
plt.title("BMI Histogram")
plt.show()
sns.boxplot(data=df['bmi'])
plt.title("BMI Boxplot")
plt.show()

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
da = (df['bmi'] < (Q1 - 1.5 * IQR)) | (df['bmi'] > (Q3 + 1.5 * IQR))
print("BMI outlier counts:\n", da.value_counts())
print("% NULL values in bmi:", df['bmi'].isna().sum() / len(df['bmi']) * 100)

df_na = df.loc[df['bmi'].isnull()]
g = df_na['stroke'].sum()
h = df['stroke'].sum()
print("People who got stroke and their BMI is NA:", g)
print("People who got stroke and their BMI is given:", h)
print("% of people with stroke in NaN values:", g / h * 100)

# Impute BMI null values with median (cannot use mean — outliers present)
print("Median of BMI:", df['bmi'].median())
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Age analysis
sns.histplot(data=df['age'])
plt.title("Age Histogram")
plt.show()
sns.boxplot(data=df['age'])
plt.title("Age Boxplot")
plt.show()

# Average glucose level analysis
sns.histplot(data=df['avg_glucose_level'])
plt.title("Avg Glucose Level Histogram")
plt.show()
sns.boxplot(data=df['avg_glucose_level'])
plt.title("Avg Glucose Level Boxplot")
plt.show()

Q1 = df['avg_glucose_level'].quantile(0.25)
Q3 = df['avg_glucose_level'].quantile(0.75)
IQR = Q3 - Q1
da = (df['avg_glucose_level'] < (Q1 - 1.5 * IQR)) | (df['avg_glucose_level'] > (Q3 + 1.5 * IQR))
print("Avg glucose outlier counts:\n", da.value_counts())

# Correlation matrix
corrmat = df.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidth=0.8, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Heart disease analysis
print(df['heart_disease'].value_counts())
df['heart_disease'].value_counts().plot(kind="pie", title="Heart Disease")
plt.show()

# Ever married analysis
print(df['ever_married'].value_counts())
df['ever_married'].value_counts().plot(kind="pie", title="Ever Married")
plt.show()

# Cross analysis — all features vs target
sns.countplot(x='stroke', hue='gender', data=df)
plt.show()
sns.countplot(x='stroke', hue='work_type', data=df)
plt.show()
sns.countplot(x='stroke', hue='smoking_status', data=df)
plt.show()
sns.countplot(x='stroke', hue='Residence_type', data=df)
plt.show()
sns.countplot(x='stroke', hue='heart_disease', data=df)
plt.show()
sns.countplot(x='stroke', hue='ever_married', data=df)
plt.show()

# Distribution plots
fig, axs = plt.subplots(2, 3, figsize=(23, 12))
sns.histplot(df['gender'], kde=False, ax=axs[0, 0])
sns.histplot(df['age'], kde=False, ax=axs[0, 1])
sns.histplot(df['work_type'], kde=False, ax=axs[0, 2])
sns.histplot(df['avg_glucose_level'], kde=False, ax=axs[1, 0])
sns.histplot(df['bmi'], kde=False, ax=axs[1, 1])
sns.histplot(df['smoking_status'], kde=False, ax=axs[1, 2])
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
sns.boxplot(x=df["age"], ax=axs[0])
sns.boxplot(x=df["avg_glucose_level"], ax=axs[1])
sns.boxplot(x=df["bmi"], ax=axs[2])
plt.show()

# ─── LABEL ENCODING ──────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

categorical_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical_col:
    df[col] = le.fit_transform(df[col])

# ─── DUMMY VARIABLES ─────────────────────────────────────────────────────────
df[['hypertension', 'heart_disease', 'stroke']] = df[['hypertension', 'heart_disease', 'stroke']].astype(str)
df = pd.get_dummies(df, drop_first=True)
print(df.head())

# ─── OVERSAMPLING ────────────────────────────────────────────────────────────
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
X = df.drop(['stroke_1'], axis=1)
y = df['stroke_1']
X_over, y_over = oversample.fit_resample(X, y)

# ─── FEATURE SCALING ─────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
df[['bmi', 'avg_glucose_level', 'age']] = s.fit_transform(df[['bmi', 'avg_glucose_level', 'age']])

# ─── SMOTE ───────────────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X, y = smote.fit_resample(df.loc[:, df.columns != 'stroke_1'], df['stroke_1'])
print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))

# ─── TRAIN-TEST SPLIT (80-20) ────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=42)
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# ─── MODEL TRAINING ──────────────────────────────────────────────────────────

from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score, roc_curve)

# ── Decision Tree ─────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_prob_dt = clf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve of DECISION TREE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_).plot()
plt.show()

print("[INFO] evaluating after training...")
print("!!Confusion Matrix Report for Decision Tree!!")
print(classification_report(y_test, y_pred))
print("Accuracy_score:", '{:.4%}'.format(accuracy_score(y_test, y_pred)))
print("ROC AUC Score:", '{:.4%}'.format(roc_auc_score(y_test, y_pred_prob_dt)))

# ── KNN ───────────────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_knn)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve of KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

cm = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_).plot()
plt.show()

print("[INFO] evaluating after training...")
print("!!Confusion Matrix Report for KNN!!")
print(classification_report(y_test, y_pred_knn))
print("Accuracy_score:", '{:.4%}'.format(accuracy_score(y_test, y_pred_knn)))
print("ROC AUC Score:", '{:.4%}'.format(roc_auc_score(y_test, y_pred_prob_knn)))

# ── XGBoost ───────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_prob_xgb)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve of XGBOOST')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

cm = confusion_matrix(y_test, y_pred_xgb, labels=xgb.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb.classes_).plot()
plt.show()

print("[INFO] evaluating after training...")
print("!!Confusion Matrix Report for XGBoost!!")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy_score:", '{:.4%}'.format(accuracy_score(y_test, y_pred_xgb)))
print("ROC AUC Score:", '{:.4%}'.format(roc_auc_score(y_test, y_pred_prob_xgb)))

# ── Random Forest ─────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_pred_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve of RANDOM FOREST')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

cm = confusion_matrix(y_test, y_pred_rf, labels=rf_clf.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_clf.classes_).plot()
plt.show()

print("[INFO] evaluating after training...")
print("!!Confusion Matrix Report for Random Forest!!")
print(classification_report(y_test, y_pred_rf))
print("Accuracy_score:", '{:.4%}'.format(accuracy_score(y_test, y_pred_rf)))
print("ROC AUC Score:", '{:.4%}'.format(roc_auc_score(y_test, y_pred_prob_rf)))

# ── Logistic Regression ───────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred_lr = classifier.predict(X_test)
y_pred_prob_lr = classifier.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_prob_lr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve of LOGISTIC REGRESSION')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

cm = confusion_matrix(y_test, y_pred_lr, labels=classifier.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_).plot()
plt.show()

print("[INFO] evaluating after training...")
print("!!Confusion Matrix Report for Logistic Regression!!")
print(classification_report(y_test, y_pred_lr))
print("Accuracy_score:", '{:.4%}'.format(accuracy_score(y_test, y_pred_lr)))
print("ROC AUC Score:", '{:.4%}'.format(roc_auc_score(y_test, y_pred_prob_lr)))

# ─── K-FOLD CROSS VALIDATION ─────────────────────────────────────────────────
from sklearn import model_selection
from sklearn.model_selection import KFold

kfold = model_selection.KFold(n_splits=20, shuffle=True)

for name, model in [("Logistic Regression", classifier),
                    ("Random Forest", rf_clf),
                    ("XGBoost", xgb),
                    ("Decision Tree", clf)]:
    scores = model_selection.cross_val_score(model, X, y, cv=kfold)
    print(f"\n{name} K-Fold Scores: {scores}")
    print(f"Average: {'{:.4%}'.format(scores.mean())}")

# ─── SAMPLE PREDICTION ───────────────────────────────────────────────────────
input_features = [80, 105.92, 32.5, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
features_name = ['age', 'avg_glucose_level', 'bmi',
                 'gender_Male', 'hypertension_1', 'heart_disease_1',
                 'ever_married_Yes', 'work_type_Never_worked',
                 'work_type_Private', 'work_type_Self_employed',
                 'work_type_children', 'Residence_type_Urban',
                 'smoking_status_formerly_smoked',
                 'smoking_status_never_smoked', 'smoking_status_smokes']

df_pred = pd.DataFrame([np.array(input_features)], columns=features_name)
prediction = rf_clf.predict(df_pred)[0]

if prediction == 1:
    print('Result: Cerebrovascular Accident Detected')
elif prediction == 0:
    print('Result: Cerebrovascular Accident Not Detected')
else:
    print('**Insufficient Information**')

# ─── SAVE MODEL ──────────────────────────────────────────────────────────────
import pickle
import os

os.makedirs('models', exist_ok=True)
with open('models/model.pickle', 'wb') as f:
    pickle.dump(rf_clf, f)
print("Model saved to models/model.pickle")
