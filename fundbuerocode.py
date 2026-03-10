import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import uuid

# --- KONFIGURATION ---
st.set_page_config(page_title="Fundstück-Scanner & Sammler", layout="centered")

# Verzeichnis für die gesammelten Daten
DATASET_DIR = "gesammelte_daten"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

@st.cache_resource
def load_keras_model():
    return load_model("keras_model.h5", compile=False)

def predict(image_data, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return index, class_names[index], prediction[0][index]

# --- UI DESIGN ---
st.title("🔍 Fundstück-Erkennung & Training")

# Modell und Labels laden
model = load_keras_model()
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Datei-Upload
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Hochgeladenes Foto', use_container_width=True)
    
    # KI-ANALYSE
    st.subheader("1. KI-Analyse")
    idx, label, score = predict(image, model, class_names)
    clean_label = label[2:] if label[0].isdigit() else label
    
    if score > 0.7:
        st.success(f"KI sagt: **{clean_label}** ({score:.2%})")
    else:
        st.warning(f"KI ist unsicher: Vermutlich {clean_label} ({score:.2%})")

    st.write("---")

    # MANUELLE KLASSIFIZIERUNG (DATEN SAMMELN)
    st.subheader("2. Manuelle Klassifizierung / Speichern")
    st.write("Hilf der KI, indem du das Bild der richtigen Gruppe zuordnest:")
    
    # Dropdown zur Auswahl der richtigen Klasse
    selected_class = st.selectbox("In welche Gruppe gehört dieses Fundstück?", class_names)
    
    if st.button("Bild in dieser Gruppe speichern"):
        # Ordner für die gewählte Klasse erstellen
        target_dir = os.path.join(DATASET_DIR, selected_class.replace(" ", "_"))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Eindeutigen Dateinamen generieren
        file_name = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(target_dir, file_name)
        
        # Bild speichern
        image.save(file_path)
        st.balloons()
        st.success(f"Gespeichert unter: `{file_path}`")

# --- SIDEBAR ---
st.sidebar.header("Statistik")
if os.path.exists(DATASET_DIR):
    for folder in os.listdir(DATASET_DIR):
        count = len(os.listdir(os.path.join(DATASET_DIR, folder)))
        st.sidebar.write(f"📁 {folder}: {count} Bilder")
