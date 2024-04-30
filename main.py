import speech_recognition as srec
import pyttsx3 as pyt
import tensorflow as tf
import numpy as np
from gtts import gTTS
import pywhatkit
import wikipedia

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
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_miki():
    dengar = perintah()
    ngomong(dengar)
    print(dengar)
    
    # Klasifikasi menggunakan model RNNs
    encoded_dengar = tokenizer.texts_to_sequences([dengar])
    padded_dengar = tf.keras.preprocessing.sequence.pad_sequences(encoded_dengar, maxlen=max_length, padding='post')
    predicted_category = np.argmax(model.predict(padded_dengar), axis=-1)
    
    category = ['buka', 'cari', 'tutup'][predicted_category[0]]

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

    else:
        pyt.speak('maaf saya tidak bisa mendengar anda, tolong ulangi')

# Persiapan data training dan preprocessing teks
commands = ['buka', 'cari', 'tutup']
train_data = [
    ('buka youtube', 'buka'),
    ('cari lagu', 'cari'),
    ('tutup aplikasi', 'tutup')
    # Tambahkan contoh-contoh lainnya di sini
]
texts = [text for text, _ in train_data]
labels = [commands.index(label) for _, label in train_data]

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

# Menjalankan miki
while True:
    run_miki()
