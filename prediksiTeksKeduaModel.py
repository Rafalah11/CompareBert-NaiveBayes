import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Muat model Naïve Bayes yang telah disimpan
with open("models/naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Muat vectorizer TF-IDF yang telah disimpan
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer_tfidf = pickle.load(f)

# Muat tokenizer dan model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Melakukan prediksi dengan model Naïve Bayes
test_texts = ["Sample text for prediction"]  # Ganti dengan teks yang ingin diprediksi
nb_predictions = nb_model.predict(vectorizer_tfidf.transform(test_texts))

print("Naïve Bayes Predictions:", nb_predictions)

# Contoh teks
test_texts = [
    "I love flying with Virgin America, the service was amazing!",
    "The flight was delayed for 3 hours, worst experience ever.",
    "The food was okay, but the seats were uncomfortable."
]

# Prediksi dengan Naïve Bayes
nb_predictions = nb_model.predict(vectorizer_tfidf.transform(test_texts))

# Prediksi dengan BERT
bert_predictions = []
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    bert_predictions.append(pred)

# Mapping label numerik ke sentimen
sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
bert_predictions = [sentiment_map[pred] for pred in bert_predictions]

# Tampilkan hasil
print("Contoh Prediksi:")
for i, text in enumerate(test_texts):
    print(f"\nText: {text}")
    print(f"Naïve Bayes Prediction: {nb_predictions[i]}")
    print(f"BERT Prediction: {bert_predictions[i]}")
