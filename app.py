from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import os
import io
import base64
import pickle
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === Konfigurasi ===
app = Flask(__name__)
MODEL_PATH = 'model/lstm_model.h5'
TOKENIZER_PATH = 'model/tokenizer.pkl'
DATA_PATH = 'indonesiagelap_clean.csv'
MAX_LEN = 100
NUM_WORDS = 5000
LABEL_MAP = {0: 'negatif', 1: 'netral', 2: 'positif'}

# === Siapkan folder model jika belum ada ===
os.makedirs('model', exist_ok=True)

# === Preprocessing tools ===
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    text = ' '.join(text.split())
    return text

# === Load atau Latih Model ===
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    print("Melatih model baru...")
    df = pd.read_csv(DATA_PATH)
    texts = df["komentar"].astype(str).apply(clean_text).tolist()
    labels = df["sentimen"].astype(str).tolist()

    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    y_cat = to_categorical(y, num_classes=3)

    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    model = Sequential()
    model.add(Embedding(input_dim=NUM_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded, y_cat, epochs=3, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)

else:
    print("Model ditemukan. Memuat model...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

# === Load Data Awal ===
df_global = pd.read_csv(DATA_PATH)

# === Fungsi Prediksi ===
def predict_sentiment(texts):
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    preds = model.predict(padded)
    labels = [LABEL_MAP[p.argmax()] for p in preds]
    return labels

# === Fungsi Wordcloud ===
def generate_wordcloud(texts):
    text = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.read()).decode('utf-8')

@app.route("/")
def index():
    return render_template("index.html", tables=[df_global.head(20).to_html(classes='table table-striped')])

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        df_new = pd.read_csv(file)
        df_new["sentimen"] = predict_sentiment(df_new["komentar"].astype(str))
        df_new.to_csv("hasil_klasifikasi.csv", index=False)
        return render_template("index.html", tables=[df_new.head(20).to_html(classes='table table-striped')])
    return render_template("upload.html")

@app.route("/visuals")
def visuals():
    df = df_global.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    fig1 = px.histogram(df, x="sentimen", title="Distribusi Sentimen")
    fig2 = px.histogram(df, x="date", title="Aktivitas Komentar per Hari")
    fig3 = px.histogram(df, x="hour", title="Aktivitas Komentar per Jam")

    graphs = [fig1.to_html(full_html=False), fig2.to_html(full_html=False), fig3.to_html(full_html=False)]
    return render_template("visuals.html", graphs=graphs)

@app.route("/wordclouds")
def wordclouds():
    pos_img = generate_wordcloud(df_global[df_global["sentimen"] == "Positif"]["komentar"])
    neg_img = generate_wordcloud(df_global[df_global["sentimen"] == "Negatif"]["komentar"])
    net_img = generate_wordcloud(df_global[df_global["sentimen"] == "Netral"]["komentar"])
    return render_template("wordclouds.html", pos_img=pos_img, neg_img=neg_img, net_img=net_img)

if __name__ == "__main__":
    app.run(debug=True)
