# 📰 Fake News Detection Using Machine Learning

A Streamlit-based web application that detects whether a news article is **Fake** or **Real** using Natural Language Processing and a Machine Learning classifier (PassiveAggressiveClassifier).

---

## 📌 Project Overview

Fake news has become a significant issue in the modern digital world. This project aims to classify news articles as either **real** or **fake** using text classification techniques. The solution uses:

- **TF-IDF vectorization** for feature extraction.
- **PassiveAggressiveClassifier** for model training.
- **Streamlit** for deploying the app.

---

## 📁 Dataset

The dataset used is a combination of:

- [`Fake.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [`True.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Each file contains news articles labeled as either fake or real.

---

## 🧠 Machine Learning Model

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier**: Passive Aggressive Classifier (fast, memory-efficient for binary classification)

The model is trained on a balanced dataset and achieves high accuracy on unseen data.

---

## 🛠 Features

✅ Classifies news as **Fake** or **Real**  
✅ Interactive Streamlit UI  
✅ Clean and intuitive layout  
✅ Pretrained model loaded for instant predictions  
✅ Real-time feedback

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detection-using-ml.git
cd fake-news-detection-using-ml
```
## Install dependencies:

```bash
pip install -r requirements.txt
```
## Run the app:

```bash
streamlit run app.py
```

## 🖼 App UI Preview

![Fake News Detection App Preview](screenshots/app-preview.png)
=======
![Fake News Detection App Preview](SS1.png)
![Fake News Detection App Preview](SS2.png)

## 📈 Results
# Accuracy: ~97%

## Model: PassiveAggressiveClassifier

# Vectorization: TF-IDF

###👨‍💻 Author
```bash
Made with ❤️ by Rimjhim Mishra
```

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
<<<<<<< HEAD

=======
>>>>>>> 7298a17 (your update message)
