import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save():
    print("📰 Fake News Detection Model Training")
    print("-" * 40)
    
    # Paths
    fake_path = "Fake.csv"
    true_path = "True.csv"
    model_path = "passive_aggressive_model.joblib"
    vectorizer_path = "tfidf_vectorizer.joblib"
    
    # Check if CSV files exist
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("❌ Error: Datasets Fake.csv or True.csv not found in the directory.")
        print("Please run 'python download_assets.py' or download the datasets first.")
        return
        
    print("1. Loading datasets...")
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    
    fake["label"] = 0
    true["label"] = 1
    
    df = pd.concat([fake, true])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Loaded {len(df)} total news articles.")
    
    print("2. Splitting dataset into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )
    
    print("3. Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    print(f"   Vocabulary size: {x_train_vec.shape[1]} words.")
    
    print("4. Training PassiveAggressiveClassifier...")
    model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    model.fit(x_train_vec, y_train)
    
    # Evaluate
    predictions = model.predict(x_test_vec)
    acc = accuracy_score(y_test, predictions)
    print(f"✅ Training completed. Test Accuracy: {acc*100:.2f}%")
    
    print("5. Saving model and vectorizer assets...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"💾 Saved model to: {model_path} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")
    print(f"💾 Saved vectorizer to: {vectorizer_path} ({os.path.getsize(vectorizer_path) / (1024*1024):.2f} MB)")
    print("🎉 Done!")

if __name__ == "__main__":
    train_and_save()
