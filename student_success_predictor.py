import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# Generate Sample Dataset
# -----------------------------
np.random.seed(42)

n = 50

study_hours = np.random.randint(1, 10, n)
attendance = np.random.randint(50, 100, n)
assignment = np.random.randint(40, 100, n)

passed = []

for i in range(n):
    score = (
        study_hours[i] * 0.4 +
        attendance[i] * 0.3 +
        assignment[i] * 0.3
    )

    if score >= 40:
        passed.append(1)
    else:
        passed.append(0)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment,
    "Passed": passed
})

# -----------------------------
# Features and Target
# -----------------------------

X = df[["Study_Hours", "Attendance", "Assignment_Score"]]
y = df["Passed"]

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("="*45)
print(" VIT Student Success Predictor ")
print("="*45)

print(f"Model Accuracy : {accuracy*100:.2f}%")

print("\nEnter Student Details")

hours = float(input("Daily Study Hours : "))
attendance = float(input("Attendance (%) : "))
assignment = float(input("Previous Assignment Score : "))

user = scaler.transform([[hours, attendance, assignment]])

probability = model.predict_proba(user)[0][1] * 100

result = model.predict(user)[0]

print("\n----------- RESULT -----------")

print(f"Success Probability : {probability:.2f}%")

if result == 1:
    print("Prediction : PASS")
    print("\nAI Suggestions")

    if hours < 3:
        print("- Increase study time to 4-5 hours/day.")

    if attendance < 80:
        print("- Improve attendance above 80%.")

    if assignment < 70:
        print("- Practice previous assignments.")

    if hours >= 4 and attendance >= 80 and assignment >= 70:
        print("- Excellent! Keep maintaining your performance.")

else:
    print("Prediction : FAIL")

    print("\nAI Recommendations")

    if hours < 4:
        print("- Increase daily study hours.")

    if attendance < 75:
        print("- Attend classes regularly.")

    if assignment < 70:
        print("- Improve assignment performance.")

    print("- Revise weak subjects every day.")
    print("- Take mock tests every week.")

print("\nThank you for using Student Success Predictor.")
