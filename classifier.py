from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from data_processing import processed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Train-test split
X_train, y_train, X_test, y_test = processed('data/GTZAN/features_30_sec.csv')

# Normalize features (important for SVC and linear models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Classifier
svc_model = SVC(kernel='linear', C=1.0, random_state=42)  # Use 'linear' or 'rbf' kernel
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_test)
print("SVC Accuracy:", accuracy_score(y_test, svc_predictions))

# Linear Classifier (Logistic Regression)
lr_model = LogisticRegression(random_state=42, max_iter=1000)  # Max iterations for convergence
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
