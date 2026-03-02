import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image, ImageOps
import datetime
from supabase import create_client, Client

# ==============================
# Supabase Verbindung
# ==============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Fundstück-Erkennung", layout="centered")

st.title("🔍 Fundstück-Erkennung mit KI")

# ==============================
# Modell laden (nur einmal)
# ==============================
@st.cache_resource
def load_teachable_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_teachable_model()

# ==============================
# Bild hochladen
# ==============================
uploaded_file = st.file_uploader("📸 Lade ein Bild deines Fundstücks hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ==============================
    # Vorhersage
    # ==============================
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip().split(" ", 1)[1]
    confidence_score = float(prediction[0][index])

    st.success(f"🧠 Erkanntes Fundstück: **{class_name}**")
    st.write(f"📊 Sicherheit: {round(confidence_score * 100, 2)} %")

    # ==============================
    # Speichern in Supabase
    # ==============================
    if st.button("💾 Fundstück speichern"):

        new_entry = {
            "datum": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fundstueck": class_name,
            "confidence": round(confidence_score * 100, 2)
        }

        response = supabase.table("funde").insert(new_entry).execute()

        if response.data:
            st.success("✅ Fundstück erfolgreich in Supabase gespeichert!")
        else:
            st.error("❌ Fehler beim Speichern in Supabase.")

# ==============================
# Gespeicherte Fundstücke anzeigen
# ==============================
st.divider()
st.subheader("📂 Meine gespeicherten Fundstücke (Supabase)")

response = supabase.table("funde").select("*").order("datum", desc=True).execute()

if response.data:

    df = pd.DataFrame(response.data)
    st.dataframe(df)

    st.subheader("🔎 Habe ich ein bestimmtes Fundstück gefunden?")
    suche = st.text_input("Fundstück eingeben")

    if suche:
        treffer = df[df["fundstueck"].str.contains(suche, case=False)]

        if not treffer.empty:
            st.success("✅ Ja! Dieses Fundstück wurde gefunden:")
            st.dataframe(treffer)
        else:
            st.error("❌ Dieses Fundstück wurde noch nicht gefunden.")

else:
    st.info("Noch keine Fundstücke in Supabase gespeichert.")
