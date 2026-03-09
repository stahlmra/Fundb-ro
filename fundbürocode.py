import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image, ImageOps
import datetime
import uuid
from supabase import create_client

# ==============================
# Supabase Verbindung
# ==============================

SUPABASE_URL = "https://lnbcyhrlnyxoyravabxl.supabase.co"
SUPABASE_KEY = "sb_publishable_ihBm0N-affEABVJ20Jz5XQ_b2elhSvw"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==============================
# Streamlit Setup
# ==============================

st.set_page_config(page_title="Fundstück-Erkennung", layout="centered")

st.title("🔍 Fundstück-Erkennung mit KI")

# ==============================
# Modell laden
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

    # ==============================
    # Bild vorbereiten
    # ==============================

    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ==============================
    # KI Vorhersage
    # ==============================

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index].strip().split(" ", 1)[1]
    confidence_score = float(prediction[0][index])

    st.success(f"🧠 Erkanntes Fundstück: **{class_name}**")
    st.write(f"📊 Sicherheit: {round(confidence_score * 100, 2)} %")

    # ==============================
    # Fundstück speichern
    # ==============================

    if st.button("💾 Fundstück speichern"):

        # eindeutiger Dateiname
        file_name = f"{uuid.uuid4()}.png"

        # Bild in Supabase Bucket hochladen
        supabase.storage.from_("fundbilder").upload(
            file_name,
            uploaded_file.getvalue(),
            {"content-type": "image/png"}
        )

        # Öffentliche Bild URL
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/fundbilder/{file_name}"

        new_entry = {
            "datum": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fundstueck": class_name,
            "confidence": round(confidence_score * 100, 2),
            "bild_url": image_url
        }

        supabase.table("funde").insert(new_entry).execute()

        st.success("✅ Fundstück und Bild gespeichert!")

# ==============================
# Gespeicherte Fundstücke anzeigen
# ==============================

st.divider()
st.subheader("📂 Gespeicherte Fundstücke")

response = supabase.table("funde").select("*").order("datum", desc=True).execute()

if response.data:

    df = pd.DataFrame(response.data)

    for index, row in df.iterrows():

        st.subheader(row["fundstueck"])
        st.write(f"📊 Sicherheit: {row['confidence']} %")
        st.write(f"📅 Datum: {row['datum']}")

        if "bild_url" in row and row["bild_url"]:
            st.image(row["bild_url"])

        st.divider()

else:
    st.info("Noch keine Fundstücke gespeichert.")

# ==============================
# Suche
# ==============================

st.subheader("🔎 Fundstück suchen")

suche = st.text_input("Fundstück eingeben")

if suche:

    response = supabase.table("funde").select("*").ilike("fundstueck", f"%{suche}%").execute()

    if response.data:

        st.success("✅ Fundstücke gefunden")

        for item in response.data:

            st.write(item["fundstueck"])
            st.write(item["datum"])
            st.write(f"Sicherheit: {item['confidence']} %")

            if item["bild_url"]:
                st.image(item["bild_url"])

            st.divider()

    else:
        st.error("❌ Dieses Fundstück wurde noch nicht gefunden.")
