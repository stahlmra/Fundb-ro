import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import uuid
import shutil
from datetime import datetime

# --- KONFIGURATION ---
st.set_page_config(page_title="Fundstück-Portal Pro", layout="wide", page_icon="🕵️")

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

# --- UI ELEMENTE & DESIGN ---
st.title("🕵️ Fundstück-Portal & KI-Scanner")
st.markdown("---")

tabs = st.tabs(["📤 Fund melden", "📦 Fundgrube durchsuchen"])

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
model = load_keras_model()

# --- TAB 1: FUND MELDEN ---
with tabs[0]:
    st.header("✨ Neuen Fund registrieren")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📸 Foto des Fundstücks", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True, caption="Vorschau")

    with col2:
        if uploaded_file:
            # KI Analyse
            label, score = predict(image, model, class_names)
            clean_label = label[2:] if label[0].isdigit() else label
            st.info(f"🤖 **KI-Tipp:** Das sieht aus wie: **{clean_label}**")
            
            # Zusatzinfos abfragen
            selected_class = st.selectbox("📂 Kategorie bestätigen", class_names)
            fundort = st.text_input("📍 Wo wurde es gefunden?", placeholder="z.B. Stadtpark, Mensa, Gleis 4")
            funddatum = st.date_input("📅 Wann wurde es gefunden?", datetime.now())
            
            if st.button("🚀 Fundstück im System speichern"):
                # Speicher-Logik mit Metadaten im Dateinamen
                # Wir nutzen Unterstriche als Trenner für die Infos
                target_dir = os.path.join(DATASET_DIR, selected_class.replace(" ", "_"))
                os.makedirs(target_dir, exist_ok=True)
                
                # Dateiname: Datum_Ort_UUID.jpg (Sonderzeichen in Ort entfernen)
                safe_ort = fundort.replace(" ", "-").replace("_", "-")
                file_name = f"{funddatum}_{safe_ort}_{uuid.uuid4().hex[:8]}.jpg"
                path = os.path.join(target_dir, file_name)
                
                image.save(path)
                st.balloons()
                st.success("✅ Fundstück wurde erfolgreich in die Datenbank aufgenommen!")

# --- TAB 2: FUNDGRUBE ---
with tabs[1]:
    st.header("🔎 Alle Fundstücke im Überblick")
    
    all_items = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(root, file)
                category = os.path.basename(root)
                # Infos aus Dateinamen extrahieren
                parts = file.split("_")
                datum = parts[0] if len(parts) > 1 else "Unbekannt"
                ort = parts[1] if len(parts) > 2 else "Unbekannt"
                all_items.append({"path": full_path, "cat": category, "date": datum, "loc": ort})

    if not all_items:
        st.info("Aktuell sind keine Fundstücke gemeldet. 🍵")
    else:
        # Galerie
        cols = st.columns(3)
        for i, item in enumerate(all_items):
            with cols[i % 3]:
                st.markdown(f"### 📦 {item['cat']}")
                st.image(item['path'], use_column_width=True)
                st.markdown(f"**📅 Gefunden am:** {item['date']}")
                st.markdown(f"**📍 Fundort:** {item['loc'].replace('-', ' ')}")
                
                if st.button(f"🙋 Das gehört mir!", key=f"claim_{i}"):
                    target_path = os.path.join(CLAIMED_DIR, os.path.basename(item['path']))
                    shutil.move(item['path'], target_path)
                    st.toast(f"Markiert als abgeholt!", icon='🎉')
                    st.rerun()
                st.write("---")

# --- SIDEBAR ---
st.sidebar.title("📊 Statistik")
open_count = sum([len(files) for r, d, files in os.walk(DATASET_DIR)])
claimed_count = len(os.listdir(CLAIMED_DIR))

st.sidebar.metric("Offene Funde", f"📦 {open_count}")
st.sidebar.metric("Vermittelt", f"🤝 {claimed_count}")

st.sidebar.markdown("---")
st.sidebar.write("🟢 **Status:** System bereit")
