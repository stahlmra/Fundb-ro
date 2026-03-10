import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import uuid
import shutil

# --- KONFIGURATION ---
st.set_page_config(page_title="Fundstück-Portal", layout="wide")

DATASET_DIR = "gesammelte_daten"
CLAIMED_DIR = "abgeholte_stuecke"

for d in [DATASET_DIR, CLAIMED_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

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
    return class_names[index], prediction[0][index]

# --- NAVIGATION ---
tabs = st.tabs(["📤 Fund melden", "📦 Fundgrube durchsuchen"])

# Labels laden
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
model = load_keras_model()

# --- TAB 1: FUND MELDEN ---
with tabs[0]:
    st.header("Neues Fundstück scannen")
    uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)
        
        label, score = predict(image, model, class_names)
        clean_label = label[2:] if label[0].isdigit() else label
        st.info(f"KI-Vorschlag: **{clean_label}** ({score:.2%})")
        
        selected_class = st.selectbox("Kategorie bestätigen:", class_names)
        if st.button("Fundstück registrieren"):
            target_dir = os.path.join(DATASET_DIR, selected_class.replace(" ", "_"))
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, f"{uuid.uuid4()}.jpg")
            image.save(path)
            st.success("Erfolgreich registriert!")

# --- TAB 2: FUNDGRUBE ---
with tabs[1]:
    st.header("Bereits gefundene Objekte")
    
    # Alle Bilder aus den Unterordnern sammeln
    all_items = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(root, file)
                category = os.path.basename(root)
                all_items.append((full_path, category))

    if not all_items:
        st.write("Aktuell keine Fundstücke in der Datenbank.")
    else:
        # Galerie-Ansicht (3 Spalten)
        cols = st.columns(3)
        for i, (img_path, cat) in enumerate(all_items):
            with cols[i % 3]:
                st.image(img_path, use_column_width=True)
                st.caption(f"Kategorie: {cat}")
                
                # Button mit eindeutigem Key pro Bild
                if st.button(f"Das gehört mir!", key=f"claim_{i}"):
                    # Verschieben in "Abgeholt"
                    target_path = os.path.join(CLAIMED_DIR, os.path.basename(img_path))
                    shutil.move(img_path, target_path)
                    st.success("Markiert! Bitte im Fundbüro melden.")
                    st.rerun() # Seite neu laden, um Bild aus Galerie zu entfernen

# --- SIDEBAR ---
st.sidebar.title("Status")
count = sum([len(files) for r, d, files in os.walk(DATASET_DIR)])
st.sidebar.metric("Offene Fundstücke", count)
claimed_count = len(os.listdir(CLAIMED_DIR))
st.sidebar.metric("Bereits zurückgegeben", claimed_count)
