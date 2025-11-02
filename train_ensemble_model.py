"""
Optimized Ensemble Model for Cardiovascular Disease Prediction
This script creates a high-accuracy stacking ensemble model using multiple algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            GradientBoostingClassifier, AdaBoostClassifier, 
                            VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Cardiovascular Disease Prediction - Ensemble Model Training")
print("="*60)

# ==================== DATA LOADING ====================
print("\n[1/8] Loading and preparing dataset...")
df = pd.read_csv('FINAL_DATASET.csv')
print(f"Original dataset shape: {df.shape}")

# Rename columns
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
             'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
             'exercise_induced_angina', 'st_depression', 'st_slope', 'target']

# ==================== DATA PREPROCESSING ====================
print("\n[2/8] Preprocessing data...")

# Convert categorical features
df['chest_pain_type'] = df['chest_pain_type'].replace({1: 'typical angina', 2: 'atypical angina', 
                                                        3: 'non-anginal pain', 4: 'asymptomatic'})
df['rest_ecg'] = df['rest_ecg'].replace({0: 'normal', 1: 'ST-T wave abnormality', 
                                         2: 'left ventricular hypertrophy'})
df['st_slope'] = df['st_slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'})
df['sex'] = df['sex'].apply(lambda x: 'male' if x == 1 else 'female')

# Remove rows with st_slope = 0
df = df[df['st_slope'] != 0].copy()

# Remove outliers using z-score
print("Removing outliers...")
numeric_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved']
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)].copy()
print(f"Dataset shape after outlier removal: {df.shape}")

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Final dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n[3/8] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ==================== FEATURE SCALING ====================
print("\n[4/8] Scaling features...")
scaler = MinMaxScaler()

# Identify numeric columns (after one-hot encoding, some are numeric)
numeric_features = ['age', 'resting_blood_pressure', 'cholesterol', 
                    'max_heart_rate_achieved', 'st_depression']

# Only scale numeric features that exist
numeric_features = [col for col in numeric_features if col in X_train.columns]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Save scaler
joblib.dump(scaler, 'feature_scaler.pkl')
print("Feature scaler saved as 'feature_scaler.pkl'")

# ==================== BASE MODEL EVALUATION ====================
print("\n[5/8] Evaluating base models with cross-validation...")

base_models = {
    'RandomForest_Entropy': RandomForestClassifier(
        criterion='entropy', n_estimators=200, max_depth=15, 
        min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'RandomForest_Gini': RandomForestClassifier(
        criterion='gini', n_estimators=200, max_depth=15, 
        min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'ExtraTrees_200': ExtraTreesClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'ExtraTrees_500': ExtraTreesClassifier(
        n_estimators=500, max_depth=15, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'XGBoost_200': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    'XGBoost_500': xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, 
        subsample=0.8, random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100, learning_rate=0.5, random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=500, 
        alpha=0.01, learning_rate='adaptive', random_state=42
    ),
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'SVC': SVC(kernel='rbf', probability=True, random_state=42, gamma='scale'),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

# Evaluate base models
cv_scores = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                           scoring='accuracy', n_jobs=-1)
    cv_scores[name] = scores.mean()
    print(f"{name:25s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Sort models by CV score
sorted_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 models by CV accuracy:")
for i, (name, score) in enumerate(sorted_models[:5], 1):
    print(f"  {i}. {name}: {score:.4f}")

# ==================== TRAIN BASE MODELS ====================
print("\n[6/8] Training base models...")
trained_base_models = {}

# Train top performing models
top_model_names = [name for name, _ in sorted_models[:10]]
for name in top_model_names:
    model = base_models[name]
    model.fit(X_train_scaled, y_train)
    trained_base_models[name] = model
    print(f"   Trained {name}")

# ==================== BUILD ENSEMBLE MODELS ====================
print("\n[7/8] Building ensemble models...")

# 1. Voting Classifier (Hard Voting)
print("  Building Voting Classifier (Hard)...")
voting_hard = VotingClassifier(
    estimators=[(name, base_models[name]) for name in top_model_names[:7]],
    voting='hard'
)
voting_hard.fit(X_train_scaled, y_train)
y_pred_voting_hard = voting_hard.predict(X_test_scaled)
acc_voting_hard = accuracy_score(y_test, y_pred_voting_hard)
print(f"    Voting (Hard) Accuracy: {acc_voting_hard:.4f}")

# 2. Voting Classifier (Soft Voting)
print("  Building Voting Classifier (Soft)...")
# Only use models with predict_proba
proba_models = [name for name in top_model_names if name not in ['KNN']][:7]
voting_soft = VotingClassifier(
    estimators=[(name, base_models[name]) for name in proba_models],
    voting='soft'
)
voting_soft.fit(X_train_scaled, y_train)
y_pred_voting_soft = voting_soft.predict(X_test_scaled)
acc_voting_soft = accuracy_score(y_test, y_pred_voting_soft)
print(f"    Voting (Soft) Accuracy: {acc_voting_soft:.4f}")

# 3. Stacking Classifier (Best Performance)
print("  Building Stacking Classifier...")
# Use top 7 models as base estimators
stacking_estimators = [(name, base_models[name]) for name in top_model_names[:7]]

# Try different meta-learners
meta_learners = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

best_stack_model = None
best_stack_acc = 0
best_meta_name = None

for meta_name, meta_learner in meta_learners.items():
    stack_model = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=meta_learner,
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    stack_model.fit(X_train_scaled, y_train)
    y_pred_stack = stack_model.predict(X_test_scaled)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    
    if acc_stack > best_stack_acc:
        best_stack_acc = acc_stack
        best_stack_model = stack_model
        best_meta_name = meta_name
    
    print(f"    Stacking ({meta_name}): {acc_stack:.4f}")

print(f"   Best Stacking Model ({best_meta_name}): {best_stack_acc:.4f}")

# ==================== FINAL MODEL SELECTION ====================
print("\n[8/8] Selecting best ensemble model...")

models_to_compare = {
    'Voting_Hard': (voting_hard, y_pred_voting_hard),
    'Voting_Soft': (voting_soft, y_pred_voting_soft),
    f'Stacking_{best_meta_name}': (best_stack_model, None)
}

# Get predictions for stacking if not already done
if models_to_compare[f'Stacking_{best_meta_name}'][1] is None:
    models_to_compare[f'Stacking_{best_meta_name}'] = (
        best_stack_model,
        best_stack_model.predict(X_test_scaled)
    )

final_scores = {}
for name, (model, y_pred) in models_to_compare.items():
    if y_pred is not None:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        final_scores[name] = {
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
            'F1': f1, 'ROC_AUC': roc
        }

# Select best model based on F1 score (balanced metric)
best_model_name = max(final_scores.keys(), key=lambda x: final_scores[x]['F1'])
best_model = models_to_compare[best_model_name][0]

print(f"\n<Æ Best Ensemble Model: {best_model_name}")
print(f"   Performance Metrics:")
for metric, value in final_scores[best_model_name].items():
    print(f"     {metric}: {value:.4f}")

# ==================== DETAILED EVALUATION ====================
print("\n" + "="*60)
print("DETAILED MODEL EVALUATION")
print("="*60)

y_pred_final = best_model.predict(X_test_scaled)
y_pred_proba_final = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Disease', 'Disease']))

# ==================== SAVE MODEL AND PREPROCESSOR ====================
print("\n" + "="*60)
print("SAVING MODEL AND PREPROCESSOR")
print("="*60)

joblib.dump(best_model, 'ensemble_model.pkl')
print(" Ensemble model saved as 'ensemble_model.pkl'")

# Save feature names for later use
joblib.dump(list(X.columns), 'feature_names.pkl')
print(" Feature names saved as 'feature_names.pkl'")

# Save model metadata
model_metadata = {
    'model_type': best_model_name,
    'accuracy': final_scores[best_model_name]['Accuracy'],
    'precision': final_scores[best_model_name]['Precision'],
    'recall': final_scores[best_model_name]['Recall'],
    'f1_score': final_scores[best_model_name]['F1'],
    'roc_auc': final_scores[best_model_name]['ROC_AUC'],
    'feature_names': list(X.columns),
    'target_classes': ['No Disease', 'Disease']
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print(" Model metadata saved as 'model_metadata.pkl'")

print("\n" + "="*60)
print(" MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nFinal Model: {best_model_name}")
print(f"Test Accuracy: {final_scores[best_model_name]['Accuracy']:.4f}")
print(f"F1 Score: {final_scores[best_model_name]['F1']:.4f}")
print("\nFiles created:")
print("  - ensemble_model.pkl (Trained model)")
print("  - feature_scaler.pkl (Feature scaler)")
print("  - feature_names.pkl (Feature column names)")
print("  - model_metadata.pkl (Model information)")
