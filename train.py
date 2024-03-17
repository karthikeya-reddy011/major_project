import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
# Load the dataset
data = pd.read_csv("dataset.csv")

# Split the dataset into features (X) and labels (y)
X = data['Text']
y = data['oh_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("training")
# Preprocessing and model training
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'cyberbullying_detection_model.pkl')