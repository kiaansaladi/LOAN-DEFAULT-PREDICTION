import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. DATA PREPROCESSING
# ==========================================
print("Loading and preprocessing data...")
# Replace with your local path or keeping it dynamic
file_path = "Loan_default.csv" 
df = pd.read_csv(file_path)

# Drop irrelevant columns
if 'LoanID' in df.columns:
    df.drop(columns=['LoanID'], inplace=True)
elif 'Loan ID' in df.columns:
    df.drop(columns=['Loan ID'], inplace=True)

# Encode categorical features
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']
# Standardize column naming just in case of spaces from PDF text
df.columns = df.columns.str.replace(' ', '')

for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Define features and target
X = df.drop(columns=['Default'])
y = df['Default'].values.ravel()

# Scale numerical features
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preprocessing completed.")

# ==========================================
# 2. TRAINING BASE ML MODELS
# ==========================================
print("\nTraining Base Models...")

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, log_reg.predict(X_test)):.4f}')

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}')

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.4f}')

# ==========================================
# 3. TRAINING DEEP LEARNING MODEL (ANN)
# ==========================================
print("\nTraining Artificial Neural Network...")
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = ann_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

y_pred_ann = (ann_model.predict(X_test) > 0.5).astype("int32")
print(f'ANN Model Accuracy: {accuracy_score(y_test, y_pred_ann):.4f}')

# ==========================================
# 4. ENSEMBLE LEARNING (VOTING CLASSIFIER)
# ==========================================
print("\nTraining Ensemble Model...")
ensemble_model = VotingClassifier(estimators=[
    ('log_reg', log_reg),
    ('random_forest', rf),
    ('xgboost', xgb)
], voting='hard') 

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
print(f'Ensemble Model Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}')

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
# Plot ANN Training Loss & Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ANN Training & Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ANN Training & Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('ann_performance.png')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Default', 'Default'], 
            yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Ensemble Model')
plt.tight_layout()
plt.savefig('ensemble_confusion_matrix.png')
plt.show()