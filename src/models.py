import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Setup paths
base_path = r"C:\coding cihuy\pygwalker"
data_path = os.path.join(base_path, "data", "datamahasiswa_clean.csv")
model_dir = os.path.join(base_path, "results", "models")
os.makedirs(model_dir, exist_ok=True)

print("="*70)
print(" MACHINE LEARNING - CAMPUS EARLY WARNING SYSTEM")
print("="*70)

# ========================================
# 1. LOAD & PREPARE DATA
# ========================================
print("\n LOADING DATA...")
df = pd.read_csv(data_path)

# Encode target
df['target'] = df['STATUS_KELULUSAN'].map({
    'LULUS TEPAT WAKTU': 1,
    'TEPAT WAKTU': 1,
    'TEPAT': 1,
    'TERLAMBAT': 0,
    'TELAT': 0,
    'TIDAK TEPAT WAKTU': 0
})

# Encode categorical variables
le_gender = LabelEncoder()
df['JENIS_KELAMIN_ENCODED'] = le_gender.fit_transform(df['JENIS_KELAMIN'])

le_nikah = LabelEncoder()
df['STATUS_NIKAH_ENCODED'] = le_nikah.fit_transform(df['STATUS_NIKAH'])

le_mahasiswa = LabelEncoder()
df['STATUS_MAHASISWA_ENCODED'] = le_mahasiswa.fit_transform(df['STATUS_MAHASISWA'])

# Features & Target
features = ['IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'IPS_5', 'IPS_6', 'IPS_7', 'IPS_8',
            'UMUR', 'JENIS_KELAMIN_ENCODED', 'STATUS_NIKAH_ENCODED', 'STATUS_MAHASISWA_ENCODED']

X = df[features]
y = df['target']

print(f" Data loaded: {len(df)} rows")
print(f"   Features: {len(features)}")
print(f"   Target distribution:")
print(f"   - Lulus Tepat Waktu: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"   - Terlambat: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")

# ========================================
# 2. SPLIT DATA
# ========================================
print("\n SPLITTING DATA...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f" Train set: {len(X_train)} samples")
print(f" Test set: {len(X_test)} samples")

# ========================================
# 3. FEATURE SCALING
# ========================================
print("\n FEATURE SCALING...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(" Scaler saved")

# ========================================
# 4. MODEL COMPARISON
# ========================================
print("\n TRAINING & COMPARING MODELS...")
print("-" * 70)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"\n Training {name}...")
    
    # Train model
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross validation
    if name == 'Logistic Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ========================================
# 5. SELECT BEST MODEL & HYPERPARAMETER TUNING
# ========================================
print("\n HYPERPARAMETER TUNING (Random Forest)...")
print("-" * 70)

# Best model is usually Random Forest, so we tune it
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\n Best Parameters: {best_params}")
print(f" Best CV Score: {grid_search.best_score_:.4f}")

# Predict with best model
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f" Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ========================================
# 6. SAVE BEST MODEL
# ========================================
print("\n SAVING MODELS...")

# Save best model
with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
print(" Best model saved: best_model.pkl")

# Save label encoders
with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump({
        'gender': le_gender,
        'nikah': le_nikah,
        'mahasiswa': le_mahasiswa
    }, f)
print(" Label encoders saved")

# Save feature names
with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(features, f)
print(" Feature names saved")

# Save all results for evaluation
results['Best Random Forest'] = {
    'model': best_model,
    'accuracy': best_accuracy,
    'predictions': y_pred_best,
    'best_params': best_params
}

with open(os.path.join(model_dir, 'model_results.pkl'), 'wb') as f:
    pickle.dump({
        'results': results,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': features
    }, f)
print(" Model results saved")

# ========================================
# 7. FEATURE IMPORTANCE
# ========================================
print("\n FEATURE IMPORTANCE:")
print("-" * 70)

importances = best_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# ========================================
# 8. FINAL SUMMARY
# ========================================
print("\n" + "="*70)
print(" MODEL TRAINING SUMMARY")
print("="*70)

print("\n MODEL COMPARISON:")
for name, result in results.items():
    if name == 'Best Random Forest':
        print(f"\n {name} (BEST)")
        print(f"   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"   Parameters: {result['best_params']}")
    else:
        print(f"\n   {name}")
        print(f"   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"   CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")

print("\n TOP 5 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\n MODEL TRAINING COMPLETE!")
print(f" Models saved in: {model_dir}")
print("\nNext: Run evaluasi_model.py for detailed evaluation")