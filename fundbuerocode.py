import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# --- KONFIGURATION ---
st.set_page_config(page_title="Fundstück-Scanner", layout="centered")

@st.cache_resource
def load_keras_model():
    # Lädt das Modell nur einmal und speichert es im Cache
    return load_model("keras_model.h5", compile=False)

def predict(image_data, model, class_names):
    # Vorbereitung des Bildes (224x224)
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Bild in Array umwandeln und normalisieren
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Daten für das Modell vorbereiten
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

# --- UI DESIGN ---
st.title("🔍 Fundstück-Erkennung")
st.write("Lade ein Foto hoch, um zu sehen, ob es ein bekanntes Fundstück ist.")

# Modell und Labels laden
model = load_keras_model()
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Datei-Upload
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Dein hochgeladenes Foto', use_container_width=True)
    
    st.write("---")
    st.write("### Analyse läuft...")
    
    # Vorhersage treffen
    label, score = predict(image, model, class_names)
    
    # Ergebnis anzeigen
    # Wir entfernen die ersten zwei Zeichen (z.B. "0 "), falls vorhanden
    clean_label = label[2:] if label[0].isdigit() else label
    
    if score > 0.7:  # Schwellenwert für Sicherheit (70%)
        st.success(f"**Gefunden!** Das ist ein: **{clean_label}**")
        st.info(f"Sicherheit der KI: {score:.2%}")
    else:
        st.warning("Das Fundstück konnte nicht eindeutig identifiziert werden.")
        st.write(f"Vermutung: {clean_label} ({score:.2%})")

# --- HINWEIS ---
st.sidebar.info("Stelle sicher, dass 'keras_model.h5' und 'labels.txt' im selben Ordner liegen.")
