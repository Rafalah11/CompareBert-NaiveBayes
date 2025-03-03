import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

# 1. Load dataset
df = pd.read_csv("archive/Tweets.csv")

# 2. Cek missing values
print("Jumlah data kosong per kolom:")
print(df.isnull().sum())

# Hapus data yang kosong di kolom review atau rating
df = df.dropna(subset=["airline_sentiment", "text", "airline"])

# 3. Tampilkan beberapa sampel data
print(df.head())

# 4. Cek jumlah data dan tipe data
print(df.info())

# 5. Cek distribusi rating
plt.figure(figsize=(8,5))
sns.countplot(x=df["airline_sentiment"], palette="coolwarm")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Tweet")
plt.title("Distribusi Sentimen Maskapai di Twitter")
plt.show()

df.rename(columns={"airline_sentiment": "sentiment"}, inplace=True)  # Sesuaikan nama kolom

print(df[["text", "sentiment"]].head())

# 7. Cek distribusi sentimen
plt.figure(figsize=(8,5))
sns.countplot(x=df["sentiment"], palette="coolwarm", order=["negative", "neutral", "positive"])
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Tweet")
plt.title("Distribusi Sentimen Maskapai di Twitter")
plt.show()

# 8. Membersihkan teks (hapus karakter khusus, angka, ubah ke lowercase)
def clean_text(text):
    text = text.lower()  # Ubah ke huruf kecil
    text = re.sub(r"[^a-z\s]", "", text)  # Hanya menyisakan huruf dan spasi
    return text

df["cleaned_text"] = df["text"].apply(clean_text)  # Menggunakan "text" dari dataset

# Cek hasil pembersihan teks
print(df[["text", "cleaned_text"]].head())

import nltk
from nltk.tokenize import word_tokenize

# Download resource tokenizer NLTK (hanya perlu sekali)
nltk.download('punkt_tab')

# 9. Tokenisasi (ubah teks menjadi daftar kata)
df["tokens"] = df["cleaned_text"].apply(word_tokenize)  # Menggunakan "cleaned_text" yang sudah dibersihkan

# Cek hasil tokenisasi
print(df[["cleaned_text", "tokens"]].head())

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 10. Menghapus Stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

    # Tokenisasi
df["tokens"] = df["text"].apply(word_tokenize)

# Menghapus stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]  # lowercase untuk menghindari case-sensitive

df["filtered_tokens"] = df["tokens"].apply(remove_stopwords)

# 11. Lemmatization (mengubah kata menjadi bentuk dasar)
lemmatizer = WordNetLemmatizer()
def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

df["lemmatized_tokens"] = df["filtered_tokens"].apply(lemmatize_words)

# Cek hasilnya
print(df[["text", "tokens", "filtered_tokens", "lemmatized_tokens"]].head())

#=======================================================================================
#=======================================================================================
#STEP 4 REPRESENTASI TEKS (BoW, TF-IDF, Word Embeddings) CONVERTING STRING TO NUMBER
#=======================================================================================
#=======================================================================================

#1. BAG OF WORD (OPSIONAL)

from sklearn.feature_extraction.text import CountVectorizer

# Inisialisasi CountVectorizer
vectorizer_bow = CountVectorizer()

# Transformasi teks menjadi vektor BoW
X_bow = vectorizer_bow.fit_transform(df["text"])

# Lihat ukuran matriks BoW
print("Shape dari BoW:", X_bow.shape)

# Contoh beberapa fitur (kata) yang dipakai
print("Fitur BoW:", vectorizer_bow.get_feature_names_out()[:20])


#2. TF-IDF (Term Frequency - Inverse Document Frequency) (PILIH YANG INI UNTUK NANTI TRAINING NAIVE BAYES)

from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TF-IDF Vectorizer
vectorizer_tfidf = TfidfVectorizer()

# Transformasi teks menjadi vektor TF-IDF
X_tfidf = vectorizer_tfidf.fit_transform(df["text"])

# Lihat ukuran matriks TF-IDF
print("Shape dari TF-IDF:", X_tfidf.shape)

# Contoh beberapa fitur (kata) yang dipakai
print("Fitur TF-IDF:", vectorizer_tfidf.get_feature_names_out()[:20])

#3. Word Embeddings (Word2Vec) (OPSIONAL KARENA UNTUK TRAINING BERT SUDAH ADA TRANFORMER-BASED EMBEDDINGS)

import gensim
from gensim.models import Word2Vec

# Latih model Word2Vec dari data token yang telah diproses
model_w2v = Word2Vec(sentences=df["lemmatized_tokens"], vector_size=100, window=5, min_count=2, workers=4)

# Cek representasi vektor dari kata tertentu
print("Vektor kata 'app':", model_w2v.wv["app"])

# Lihat ukuran vektor per kata
print("Dimensi vektor Word2Vec:", model_w2v.wv.vector_size)

#======================================
#======================================
#STEP 5 : MODEL TRAINING (NAÏVE BAYES)
#======================================
#======================================

#1. Bagi Data Menjadi Training & Testing Set
from sklearn.model_selection import train_test_split

# Pisahkan fitur (X) dan label (y)
X = df["text"]  # Menggunakan teks asli
y = df["sentiment"]  # Sentimen (Positive, Neutral, Negative)

# Bagi data 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))

#2. Konversi Teks ke Vektor (TF-IDF & BoW)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# BoW (Bag of Words)
vectorizer_bow = CountVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# TF-IDF (Term Frequency - Inverse Document Frequency)
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

print("BoW shape:", X_train_bow.shape)
print("TF-IDF shape:", X_train_tfidf.shape)

#3. Train Model Naïve Bayes dengan TF-IDF

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Buat model Naïve Bayes
nb_model = MultinomialNB()

# Latih model dengan TF-IDF
nb_model.fit(X_train_tfidf, y_train)

# Prediksi
y_pred = nb_model.predict(X_test_tfidf)

print("Training Model Naïve Bayes Selesai!")

# Evaluasi
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. Simpan Model Naïve Bayes dan TF-IDF Vectorizer
import pickle
import os

# Buat folder models/ jika belum ada
os.makedirs("models", exist_ok=True)

# Simpan Model Naïve Bayes
with open("models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

# Simpan TF-IDF Vectorizer
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer_tfidf, f)

#=================================
#=================================
#STEP 6: MODEL TRAINING (BERT)
#=================================
#=================================

#===========================
# STEP 1 - Load Tokenizer dan Model BERT
#===========================
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Inisialisasi tokenizer BERT dan model BERT untuk klasifikasi sentimen
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

#===========================
# STEP 2 - Konversi Label Sentimen ke Angka
#===========================
# Mempersiapkan konversi label sentimen ke angka (positive, neutral, negative)
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_label'] = df['sentiment'].map(sentiment_map)

#===========================
# STEP 3 - Tokenisasi Semua Teks
#===========================
# Tokenisasi teks
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

#===========================
# STEP 4 - Buat Dataset PyTorch
#===========================
from torch.utils.data import Dataset

# Membuat custom dataset untuk PyTorch
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sentiment_map):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.sentiment_map = sentiment_map
        self.texts = dataframe["text"].values
        self.labels = dataframe["sentiment_label"].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        encodings['labels'] = torch.tensor(label)
        return encodings

# Membuat dataset
train_dataset = SentimentDataset(df, tokenizer, sentiment_map)

#===========================
# STEP 5 - Dataloader untuk Training
#===========================
from torch.utils.data import DataLoader

# Membuat dataloader untuk training
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#===========================
# STEP 6 - Training Model BERT
#===========================
from transformers import AdamW
from tqdm import tqdm

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fungsi untuk melatih model
def train_model(model, train_dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
            loop.set_postfix(loss=loss.item())

# Transfer model ke GPU jika tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Latih model
train_model(model, train_dataloader, optimizer, num_epochs=3)

#===========================
# STEP 7 - Evaluasi Model BERT
#===========================
# Fungsi untuk evaluasi
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Menghitung akurasi model
accuracy = evaluate_model(model, train_dataloader)
print(f"Akurasi pada data training: {accuracy * 100:.2f}%")

#===========================
# STEP 7.1 - Evaluasi dengan Sklearn
#===========================
from sklearn.metrics import classification_report

# Mendapatkan prediksi dan label
all_preds = []
all_labels = []
for batch in train_dataloader:
    input_ids = batch['input_ids'].squeeze(1).to(device)
    attention_mask = batch['attention_mask'].squeeze(1).to(device)
    labels = batch['labels'].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Menampilkan classification report
print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))

#===========================
# STEP 8 - Simpan Model BERT
#===========================
# Simpan model BERT
model.save_pretrained('models/bert_sentiment_model')

# Simpan tokenizer BERTy
tokenizer.save_pretrained('models/bert_tokenizer')

