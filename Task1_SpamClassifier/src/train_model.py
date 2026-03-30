import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Incarcarea setului de date
    df = pd.read_csv("data/products.csv")

    # Curatarea numelor coloanelor
    df.columns = df.columns.str.strip()

    # Eliminarea randurilor fara titlu sau categorie
    df = df.dropna(subset=["Product Title", "Category Label"])

    # Definirea variabilelor X (titlu) si y (categorie)
    X = df["Product Title"]
    y = df["Category Label"]

    # Impartirea datelor in train si test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Construirea pipeline-ului TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000
        )),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # Antrenarea modelului
    pipeline.fit(X_train, y_train)

    # Evaluarea modelului
    y_pred = pipeline.predict(X_test)
    print("\nEvaluarea modelului:")
    print("Acuratețe:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # Salvarea modelului intr-un fisier .pkl
    os.makedirs("models", exist_ok=True)
    with open("models/product_classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("\nModelul a fost salvat in models/product_classifier.pkl")

if __name__ == "__main__":
    main()

