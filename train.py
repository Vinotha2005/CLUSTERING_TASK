# ==============================================
# Student Performance Classification (Clean Version)
# ==============================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("data_student (1).csv")

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# Step 2: Handle missing values
# -----------------------------
df.fillna(method='ffill', inplace=True)

# -----------------------------
# Step 3: Encode categorical target
# -----------------------------
if df['UNS'].dtype == 'object':
    le = LabelEncoder()
    df['UNS'] = le.fit_transform(df['UNS'])
else:
    le = None

# -----------------------------
# Step 4: Split features and target
# -----------------------------
X = df.drop('UNS', axis=1)
y = df['UNS']

# -----------------------------
# Step 5: Split train-test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 6: Scale numeric data
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 7: Train RandomForest model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Step 8: Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 9: Save model and preprocessing
# -----------------------------
pickle.dump(model, open("student_model.pkl", "wb"))
pickle.dump(scaler, open("student_scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("student_features.pkl", "wb"))
if le is not None:
    pickle.dump(le, open("student_labelencoder.pkl", "wb"))

print("\n✅ Model training completed and saved successfully!")
