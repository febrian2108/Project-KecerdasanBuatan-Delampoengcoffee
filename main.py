import speech_recognition as srec
import pyttsx3 as pyt
import tensorflow as tf
import numpy as np
from gtts import gTTS
import pywhatkit
import wikipedia
import webbrowser

engine = pyt.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def perintah():
    mendengar = srec.Recognizer()
    with srec.Microphone() as source:
        pyt.speak('hallo saya miki, ada yang bisa saya bantu?')
        print('Mendengarkan....')
        suara = mendengar.listen(source, phrase_time_limit=10)
        try: 
            print('Diterima...')
            dengar = mendengar.recognize_google(suara, language='id-ID')
            print(dengar)
            return dengar
        except srec.UnknownValueError:
            print("Maaf, saya tidak bisa mendengar apa yang Anda katakan.")
            return ""
        except srec.RequestError as e:
            print("Maaf, terjadi kesalahan pada sistem.")
            return ""

def ngomong(teks):
    bahasa = 'id'
    engine.say(teks)
    engine.runAndWait()

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(units=64),
        tf.keras.layers.Dense(10, activation='softmax')  # Mengubah jumlah kelas menjadi 10
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_jarvis():
    dengar = perintah()
    ngomong(dengar)
    print(dengar)
    
    # Klasifikasi menggunakan model RNNs
    encoded_dengar = tokenizer.texts_to_sequences([dengar])
    padded_dengar = tf.keras.preprocessing.sequence.pad_sequences(encoded_dengar, maxlen=max_length, padding='post')
    predicted_category = np.argmax(model.predict(padded_dengar), axis=-1)
    
    category = ['buka', 'cari', 'tutup', 'pesan', 'rekomendasi', 'lokasi', 'promo'][predicted_category[0]]

    if category == 'buka':
        video = dengar.replace('buka', '')
        pyt.speak('membuka ' + video)
        print(video + ' dibuka...')
        pywhatkit.playonyt(video)

    elif category == 'cari':
        search_query = dengar.replace('cari', '')
        hasil = wikipedia.summary(search_query, sentences=1)
        print(hasil)
        pyt.speak(hasil)

    elif category == 'tutup':
        pyt.speak('baik tuan')
        exit()

    elif category == 'pesan':
        pesanan = perintah()
        pyt.speak('anda bisa memesan kopi melalui link ini.')
        print('link : https://www.tokopedia.com/delampoengcoffee.')
        # menampilkan link pembelian

    elif category == 'rekomendasi':
        pyt.speak('Berikut adalah rekomendasi kopi dari delampoeng coffee.')
        pyt.speak('product best seller dari delampoeng coffe adalah.')
        pyt.speak('1. kopi robusta.')
        pyt.speak('2. kopi arabica.')
        pyt.speak('3. kopi luwak.')
        pyt.speak('4. kopi liberica.')
        pyt.speak('5. kopi excelsa.')
        pyt.speak('6. kopi decaf.')
        pyt.speak('7. blend kopi (campuran antara kopi arabica dan kopi robusta).')
        #menampilkan teks
        print('Berikut adalah rekomendasi kopi dari delampoeng coffee.')
        print('product best seller dari delampoeng coffe adalah.')
        print('1. kopi robusta.')
        print('2. kopi arabica.')
        print('3. kopi luwak.')
        print('4. kopi liberica.')
        print('5. kopi excelsa.')
        print('6. kopi decaf.')
        print('7. blend kopi (campuran antara kopi arabica dan kopi robusta).')
        # Berikan rekomendasi menu kopi kepada pengguna

    elif category == 'lokasi':
        pyt.speak('Berikut adalah lokasi Delampoeng Coffee.')
        # Berikan informasi tentang lokasi Delampoeng Coffee menggunakan layanan peta
        # Dapatkan lokasi Delampoeng Coffee menggunakan API atau sumber lain
        lokasi = 'lokasi'  # Ganti dengan lokasi yang ingin ditampilkan
        url = f"https://maps.app.goo.gl/c4oyCA7dfXjTrQ7dA={lokasi}"
        webbrowser.open(url)

    elif category == 'promo':
        pyt.speak('tidak ada promo untuk hari ini dari Delampoeng Coffee, mohon periksa lagi beberapa waktu kedepan.')
        print('tidak ada promo untuk hari ini dari Delampoeng Coffee, mohon periksa lagi beberapa waktu kedepan.')
        # Berikan informasi tentang promosi dan diskon dari Delampoeng Coffee

    else:
        pyt.speak('maaf saya tidak bisa mendengar anda, tolong ulangi')
        print('maaf saya tidak bisa mendengar anda, tolong ulangi')

# Persiapan data training dan preprocessing teks
commands = ['buka', 'cari', 'tutup', 'pesan', 'rekomendasi', 'lokasi', 'promo',]
train_data = [
    ('buka'),
    ('cari'),
    ('tutup'),
    ('pesan'),
    ('rekomendasi'),
    ('lokasi'),
    ('promo'),
    # Tambahkan contoh-contoh lainnya di sini
]
texts = train_data
labels = [commands.index(label) for label in train_data]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(text.split()) for text in texts])
embedding_dim = 16

# Membangun dan melatih model RNNs
model = build_model()
encoded_texts = tokenizer.texts_to_sequences(texts)
padded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, maxlen=max_length, padding='post')
model.fit(padded_texts, np.array(labels), epochs=10)

# Menjalankan Jarvis
while True:
    run_jarvis()
