import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import uuid
from datetime import datetime
from supabase import create_client, Client
import io

# --- SUPABASE SETUP ---
# Ersetze diese Werte mit deinen echten Daten aus Supabase (Settings -> API)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- KONFIGURATION ---
st.set_page_config(page_title="KI Fundbüro Cloud", layout="wide", page_icon="🕵️")

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

# --- UI ELEMENTE ---
st.title("🕵️ Cloud-Fundbüro mit KI")
st.markdown("---")

tabs = st.tabs(["📤 Fund melden", "🔎 Fundgrube durchsuchen"])

# Labels laden
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("Datei 'labels.txt' nicht gefunden!")
    class_names = ["Standard"]

model = load_keras_model()

# --- TAB 1: FUND MELDEN ---
with tabs[0]:
    st.header("✨ Neuen Fund in die Cloud hochladen")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📸 Foto aufnehmen oder wählen", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True, caption="Vorschau")

    with col2:
        if uploaded_file:
            label, score = predict(image, model, class_names)
            clean_label = label[2:] if label[0].isdigit() else label
            st.info(f"🤖 **KI-Tipp:** {clean_label} ({score:.2%})")
            
            selected_class = st.selectbox("📂 Kategorie bestätigen", class_names)
            fundort = st.text_input("📍 Fundort", placeholder="Wo wurde es gefunden?")
            funddatum = st.date_input("📅 Funddatum", datetime.now())
            
            if st.button("🚀 In der Cloud speichern"):
                with st.spinner("Lade hoch..."):
                    # 1. Bild in Byte-Stream umwandeln für den Upload
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    file_name = f"{uuid.uuid4()}.jpg"
                    
                    # 2. Upload in Supabase Storage
                    try:
                        storage_res = supabase.storage.from_("fundstuecke_bilder").upload(
                            path=file_name,
                            file=img_bytes,
                            file_options={"content-type": "image/jpeg"}
                        )
                        
                        # 3. Öffentliche URL generieren
                        img_url = supabase.storage.from_("fundstuecke_bilder").get_public_url(file_name)
                        
                        # 4. Eintrag in Datenbank-Tabelle
                        db_data = {
                            "kategorie": selected_class,
                            "fundort": fundort,
                            "funddatum": str(funddatum),
                            "bild_url": img_url,
                            "status": "offen"
                        }
                        supabase.table("fundstuecke").insert(db_data).execute()
                        
                        st.balloons()
                        st.success("Erfolgreich in Supabase gespeichert!")
                    except Exception as e:
                        st.error(f"Fehler beim Upload: {e}")

# --- TAB 2: FUNDGRUBE ---
with tabs[1]:
    st.header("🔎 Aktuelle Fundstücke")
    
    # Daten aus Supabase abrufen
    try:
        response = supabase.table("fundstuecke").select("*").eq("status", "offen").order("created_at", desc=True).execute()
        items = response.data

        if not items:
            st.info("Keine offenen Fundstücke gefunden.")
        else:
            cols = st.columns(3)
            for i, item in enumerate(items):
                with cols[i % 3]:
                    st.markdown(f"### 📦 {item['kategorie']}")
                    st.image(item['bild_url'], use_column_width=True)
                    st.markdown(f"📅 **Gefunden am:** {item['funddatum']}")
                    st.markdown(f"📍 **Ort:** {item['fundort']}")
                    
                    if st.button(f"🙋 Das gehört mir!", key=f"claim_{item['id']}"):
                        # Status in DB auf 'abgeholt' setzen
                        supabase.table("fundstuecke").update({"status": "abgeholt"}).eq("id", item['id']).execute()
                        st.toast("Als abgeholt markiert!", icon="🎉")
                        st.rerun()
                    st.write("---")
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")

# --- SIDEBAR ---
st.sidebar.title("📊 Cloud Statistik")
try:
    res_stats = supabase.table("fundstuecke").select("status").execute()
    all_stats = res_stats.data
    offen = len([x for x in all_stats if x['status'] == 'offen'])
    weg = len([x for x in all_stats if x['status'] == 'abgeholt'])
    
    st.sidebar.metric("Offen", f"📦 {offen}")
    st.sidebar.metric("Abgeholt", f"🤝 {weg}")
except:
    st.sidebar.write("Verbindung wird aufgebaut...")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Supabase & Streamlit")
