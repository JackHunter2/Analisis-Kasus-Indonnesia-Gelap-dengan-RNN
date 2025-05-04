from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from textblob import TextBlob
from datetime import datetime
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time
import csv
import re
import os
import string
import pandas as pd

# === Keyword-based Sentiment Contextualization ===
keywords_positif = [
    "bener banget", "fakta", "setuju", "realita", "emang ekonomi", "krisis", "turun",
    "data jelas", "buka mata", "akhirnya dibahas", "parah banget", "butuh solusi", "indonesiagelap", "indonesia gelap"
]
keywords_negatif = [
    "hoaks", "lebay", "drama", "bohong", "settingan", "baik-baik saja",
    "jangan percaya", "berita negatif", "fitnah", "provokasi", "jatuhin pemerintah"
]
keywords_netral = [
    "serius", "gimana", "sumbernya", "ikut nyimak", "semoga", "😢", "😱", "amin"
]

def analyze_sentiment(text):
    text_lower = text.lower()
    for word in keywords_positif:
        if word in text_lower:
            return "positif"
    for word in keywords_negatif:
        if word in text_lower:
            return "negatif"
    for word in keywords_netral:
        if word in text_lower:
            return "netral"

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return "positif"
    elif polarity < -0.5:
        return "negatif"
    return "netral"

# Setup pembersihan & preprocessing
stop_factory = StopWordRemoverFactory()
stem_factory = StemmerFactory()
stop_remover = stop_factory.create_stop_word_remover()
stemmer = stem_factory.create_stemmer()

def clean_text(text):
    unwanted_words = ['likesReply', 'See translation', 'Reply', 'View replies', 'likes', 
                      'w', 'wReply', 'like', 'reply', 'view', 'replies', 'translation', 'see']
    text = text.lower()
    # Hapus username dan format khusus
    text = re.sub(r'[a-zA-Z0-9_]+,[a-zA-Z0-9_]+', '', text)  # Hapus pasangan username dengan koma
    text = re.sub(r'([a-zA-Z0-9_]+)[,.]?\s*\1', '', text)  # Hapus username yang berulang
    text = re.sub(r'[a-zA-Z0-9_]+_[a-zA-Z0-9_]+', '', text)  # Hapus username dengan underscore
    text = re.sub(r'\b[a-z0-9_]+\b,', '', text)  # Hapus username yang diikuti koma
    text = re.sub(r',\s*[a-z0-9_]+\b', '', text)  # Hapus username yang didahului koma
    text = re.sub(r'\b[a-z0-9_]+\.[a-z0-9_]+\b', '', text)  # Hapus username dengan titik
    text = re.sub(r'\b[a-z0-9_]+\b(?=\s*[,."]|$)', '', text)  # Hapus username di akhir atau sebelum tanda baca
    
    # Hapus teks berwarna biru dan format khusus
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Hapus mentions
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'www\.\S+', '', text)  # Hapus www
    text = re.sub(r'instagram\.com/\S+', '', text)  # Hapus link Instagram
    text = re.sub(r'<[^>]+>', '', text)  # Hapus tag HTML
    text = re.sub(r'\d+ likes?', '', text)  # Hapus "X likes"
    text = re.sub(r'\d+ reply', '', text)  # Hapus "X reply"
    text = re.sub(r'\d+ replies', '', text)  # Hapus "X replies"
    text = re.sub(r'View \d+ replies?', '', text)  # Hapus "View X replies"
    
    # Hapus kata-kata yang tidak diinginkan
    for word in unwanted_words:
        text = text.replace(word.lower(), '')
    
    # Hapus spasi berlebih
    text = ' '.join(text.split())
    
    # Hapus stopwords
    text = stop_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)
    
    return text if len(text.strip()) >= 5 else ""

# Setup ChromeDriver
service = Service('chromedriver.exe')
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

try:
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
    })

    print("Membuka Instagram...")
    driver.get("https://www.instagram.com/ajiarchive.psd/reel/DGS-TQQzArp/")

    input("➡️  Login Instagram dulu, lalu tekan Enter...")

    print("Menunggu halaman dimuat...")
    time.sleep(10)

    try:
        view_all = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'View all') or contains(text(),'Lihat semua')]"))
        )
        view_all.click()
        time.sleep(3)
    except:
        print("Tidak ada tombol 'View all comments'.")

    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    def click_all_view_replies():
        while True:
            try:
                buttons = driver.find_elements(By.XPATH, "//span[contains(text(),'View replies') or contains(text(),'Lihat balasan')]")
                if not buttons:
                    break
                for button in buttons:
                    try:
                        driver.execute_script("arguments[0].click();", button)
                        time.sleep(1)
                    except:
                        continue
            except:
                break

    print("🔄 Membuka semua balasan komentar...")
    click_all_view_replies()

    print("Mengambil komentar...")
    selectors = ["ul._a9z6 li div._a9zs span", "ul._a9z6 li span"]
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comments_data = []

    for selector in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            print(f"✅ Selector aktif: {selector}")
            for el in elements:
                try:
                    text = el.text.strip()
                    if text and not text.isdigit():
                        cleaned = clean_text(text)
                        if cleaned:
                            sentimen = analyze_sentiment(cleaned)
                            comments_data.append({
                                "platform": "Instagram",
                                "komentar": cleaned,
                                "timestamp": timestamp_now,
                                "sentimen": sentimen
                            })
                            print(f"🔹 {cleaned} | Sentimen: {sentimen}")
                except:
                    continue
            break

    driver.quit()

    if comments_data:
        df = pd.DataFrame(comments_data)
        df = df.drop_duplicates(subset=["komentar"])
        df.to_csv("indonesiagelap_clean.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ Berhasil menyimpan {len(df)} komentar ke 'indonesiagelap_clean.csv'")
        print("📂 Lokasi file:", os.path.abspath("indonesiagelap_clean.csv"))
    else:
        print("❌ Tidak ada komentar yang valid ditemukan.")

except Exception as e:
    print(f"❗ Terjadi error: {str(e)}")
    try:
        driver.quit()
    except:
        pass
