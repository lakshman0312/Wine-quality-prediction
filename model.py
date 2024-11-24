import numpy as np
import pandas as pd
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=1),
    "GaussianNB":GaussianNB(),
}

df = pd.read_csv('data.csv')

df.dropna(subset=['quality'], inplace=True)

important_features = ["alcohol", "sulphates", "volatile acidity", "total sulfur dioxide", "density"]

X = df[important_features]
y = df['quality'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

top_models = ["Random Forest", "Logistic Regression", "GaussianNB"]
# Create a list of base estimators for the ensemble
base_estimators = [(name, model) for name, model in models.items() if name in top_models]
stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)

pickle.dump(stacking_clf, open("model.pkl", "wb"))