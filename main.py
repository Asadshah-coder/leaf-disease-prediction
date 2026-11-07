"""Plant Disease Recognition System — professional web app.

Leaf Disease Prediction Using Deep Learning (Final Year Project).

A Streamlit app that identifies plant diseases from leaf photos and
helps farmers act on the result:

- Diagnose: upload up to 3 leaf photos (or take one with the camera),
  add the farm/city location, and get the top-3 predictions with
  confidence bars, disease information and local weather.
- Dashboard: session statistics and charts.
- History & Calendar: every diagnosis is saved with its date, time
  and location; browse past diagnoses on a month calendar and read
  general seasonal disease guidance.
- Disease Library: all 38 classes with symptoms and management advice.
- About: the project, the model and its verified results.

The model file (86 MB) is not in this repo. On first run the app
downloads it from Google Drive (public link) if no local model file
exists; see ``ensure_model_file``. The model expects pixel values in
[0, 1] — ``predict_top3`` divides by 255 (the training pipeline
normalises with ``Rescaling(1.0 / 255)`` and that layer is NOT part
of the saved model, so feeding raw [0, 255] pixels saturates the
network and it predicts one class at ~100% for every image).

History lives in the session only on the free Streamlit tier — it is
cleared when the browser session ends. A database is needed for
permanent history.
"""

import calendar as calmod
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

try:
    import requests
except ImportError:  # weather is optional; the app works without it
    requests = None

from disease_info import DISEASE_INFO, SEASONAL_ACTIVE, SEASONAL_NOTES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CANDIDATES = [
    "trained_plant_disease_model.keras",
    "trained_model.keras",
]

# Google Drive file ID of the trained model (public "anyone with the
# link" download). The Streamlit Cloud demo has no model file in the
# repo (the 86 MB file is too large for GitHub), so the app downloads
# it from Drive on first run — only when no local model file exists.
MODEL_DRIVE_FILE_ID = "1ufCFp8jhSFG6MMkcgF3XJsxm98y_7AjK"

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

IMAGE_SIZE = (128, 128)

# A prediction below this confidence is shown as uncertain instead of
# being hidden (kept from the original project, which hid such rows).
CONFIDENCE_THRESHOLD = 0.80

HISTORY_LIMIT = 50

# Verified model results (retrain #2, 7 Nov 2026, n=17,572 validation
# images): validation accuracy 94.21%, macro F1 0.9417.
MODEL_STATS = {
    "Validation accuracy": "94.21%",
    "Macro F1": "0.9417",
    "Classes": "38",
    "Validation images": "17,572",
}

PAGES = ["🔍 Diagnose", "📊 Dashboard", "📅 History & Calendar",
         "📚 Disease Library", "ℹ️ About"]


# ---------------------------------------------------------------------------
# Model loading (once per session)
# ---------------------------------------------------------------------------

def find_model_file() -> Path | None:
    """Return the first existing model file, or None if neither exists."""
    here = Path(__file__).resolve().parent
    for name in MODEL_CANDIDATES:
        path = here / name
        if path.is_file():
            return path
    return None


def ensure_model_file() -> Path | None:
    """Return a local model file, downloading it from Drive if needed.

    Prefers a file already next to ``main.py``. Otherwise downloads
    the trained model from Google Drive (first run on Streamlit Cloud
    only). Returns None when no model can be found or downloaded.
    """
    path = find_model_file()
    if path is not None:
        return path
    if not MODEL_DRIVE_FILE_ID:
        return None
    try:
        import gdown
    except ImportError:
        return None
    dest = Path(__file__).resolve().parent / MODEL_CANDIDATES[0]
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}"
    with st.spinner("Downloading the trained model (86 MB) — first run only..."):
        gdown.download(url, str(dest), quiet=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    return None


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Load the trained Keras model once and cache it for the session.

    Returns (model, model_path) or (None, None) when no model file exists.
    """
    path = ensure_model_file()
    if path is None:
        return None, None
    model = tf.keras.models.load_model(str(path))
    return model, path


def predict_top3(model, image_file):
    """Predict the top 3 disease classes of one uploaded leaf image.

    Returns a list of (predicted_class, confidence) sorted by
    confidence, highest first. Pixels are normalised to [0, 1] first —
    the model was trained on normalised inputs.
    """
    image = tf.keras.preprocessing.image.load_img(
        image_file, target_size=IMAGE_SIZE
    )
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.array([input_arr])  # single image -> batch of one
    predictions = model.predict(input_arr, verbose=0)[0]
    top_idx = np.argsort(predictions)[::-1][:3]
    return [(CLASS_NAMES[i], float(predictions[i])) for i in top_idx]


def show_missing_model_message() -> None:
    """Friendly explanation shown when no trained model file is present."""
    st.error("No trained model file was found, so predictions are unavailable.")
    st.markdown(
        """
        The app looks for one of these files next to `main.py`:

        - `trained_plant_disease_model.keras` (preferred name), or
        - `trained_model.keras` (the name the original training notebook
          saved under)

        **How to get a model file:**

        1. Train one on Google Colab — follow the steps in
           `docs/COLAB_TRAINING.md` (run `python src/train.py`, which saves
           the model as `trained_plant_disease_model.keras`), then place
           the downloaded `.keras` file next to `main.py` and restart
           the app; or
        2. If you already have a trained `.keras` file, rename it to
           `trained_plant_disease_model.keras` and place it next to
           `main.py`.
        """
    )


# ---------------------------------------------------------------------------
# Weather (Open-Meteo, free, no API key) and history helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def fetch_weather(lat: float, lon: float):
    """Current weather for a latitude/longitude, or None on any failure."""
    if requests is None:
        return None
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,"
                           "precipitation,wind_speed_10m",
            },
            timeout=8,
        )
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        return {
            "temperature": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"),
            "wind": cur.get("wind_speed_10m"),
        }
    except Exception:
        return None


def init_state() -> None:
    """Create the session-state keys the app needs."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    today = date.today()
    if "cal_year" not in st.session_state:
        st.session_state["cal_year"] = today.year
        st.session_state["cal_month"] = today.month
        st.session_state["cal_day"] = today
    if "detected_lat" not in st.session_state:
        st.session_state["detected_lat"] = None
        st.session_state["detected_lon"] = None


def add_diagnosis(record: dict) -> None:
    """Append a diagnosis to the session history (newest first)."""
    history = st.session_state["history"]
    history.insert(0, record)
    st.session_state["history"] = history[:HISTORY_LIMIT]


def history_frame() -> pd.DataFrame:
    """History as a DataFrame (without image bytes)."""
    rows = []
    for rec in st.session_state["history"]:
        rows.append({
            "Date": rec["time"].strftime("%Y-%m-%d"),
            "Time": rec["time"].strftime("%H:%M"),
            "Image": rec["filename"],
            "Prediction": rec["prediction"],
            "Confidence (%)": round(rec["confidence"] * 100, 2),
            "Location": rec["location"] or "—",
            "Latitude": rec["lat"] if rec["lat"] is not None else "",
            "Longitude": rec["lon"] if rec["lon"] is not None else "",
        })
    return pd.DataFrame(rows)


def pretty_class(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato — Early Blight'."""
    info = DISEASE_INFO.get(class_name, {})
    crop = info.get("crop", class_name.split("___")[0])
    label = info.get("label", class_name.split("___")[-1].replace("_", " "))
    return f"{crop} — {label}"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
  .hero { background: linear-gradient(135deg, #14532d 0%, #166534 55%, #15803d 100%);
          color: white; padding: 1.6rem 1.8rem; border-radius: 14px;
          margin-bottom: 1.2rem; }
  .hero h1 { color: white; margin: 0 0 0.3rem 0; font-size: 1.9rem; }
  .hero p { color: #dcfce7; margin: 0; }
  .stat-chip { display: inline-block; background: rgba(255,255,255,0.16);
          color: #fff; border-radius: 999px; padding: 0.25rem 0.8rem;
          margin: 0.5rem 0.4rem 0 0; font-size: 0.85rem; }
  .result-card { border: 1px solid #d1fae5; border-left: 5px solid #16a34a;
          background: #f0fdf4; border-radius: 10px; padding: 0.9rem 1.1rem;
          margin-bottom: 0.6rem; }
  .result-card.low { border-left-color: #d97706; background: #fffbeb;
          border-color: #fde68a; }
  .result-card h3 { margin: 0 0 0.2rem 0; }
  .muted { color: #6b7280; font-size: 0.9rem; }
</style>
"""


def hero(title: str, subtitle: str, chips: list | None = None) -> None:
    """Page hero banner."""
    chip_html = "".join(
        f'<span class="stat-chip">{c}</span>' for c in (chips or [])
    )
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
        f"{chip_html}</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_diagnose(model) -> None:
    """Diagnose page: upload photos, add a location, get predictions."""
    hero(
        "🌱 Plant Disease Recognition",
        "Upload a leaf photo to identify the disease, get treatment advice "
        "and save the diagnosis with its date and location.",
        chips=["38 classes", "94.21% validation accuracy",
               "17,572 validation images"],
    )

    if model is None:
        show_missing_model_message()
        return

    st.markdown("### 1. Add leaf photos")
    uploaded = st.file_uploader(
        "Choose up to 3 images",
        type=["jpeg", "jpg", "png"],
        accept_multiple_files=True,
        help="Clear, well-lit photos of a single leaf work best.",
    )
    with st.expander("📷 Or take a photo with your camera"):
        camera_image = st.camera_input("Take a photo of the leaf")
    images = list(uploaded or [])
    if camera_image is not None:
        images.append(camera_image)
    if len(images) > 3:
        st.info("Only the first 3 images will be analysed.")
        images = images[:3]

    st.markdown("### 2. Add a location (optional)")
    loc_col1, loc_col2, loc_col3 = st.columns(3)
    with loc_col1:
        location_name = st.text_input(
            "Farm / City", placeholder="e.g. Lahore",
            help="Saved with the diagnosis and shown on the map.",
        )
    with loc_col2:
        lat_input = st.number_input(
            "Latitude", value=None, format="%.6f",
            help="Optional — used for the map and local weather.",
        )
    with loc_col3:
        lon_input = st.number_input(
            "Longitude", value=None, format="%.6f",
            help="Optional — used for the map and local weather.",
        )
    if st.button("📍 Detect my location"):
        try:
            from streamlit_js_eval import get_geolocation

            loc = get_geolocation()
            coords = (loc or {}).get("coords", {}) if isinstance(loc, dict) else {}
            if coords.get("latitude") is not None:
                st.session_state["detected_lat"] = coords["latitude"]
                st.session_state["detected_lon"] = coords["longitude"]
                st.success(
                    f"Location detected: {coords['latitude']:.4f}, "
                    f"{coords['longitude']:.4f}. It will be used below."
                )
            else:
                st.info("Location is not available yet — click the button "
                        "again after allowing location access in your browser.")
        except Exception:
            st.info("Automatic location detection is not available in this "
                    "browser — please enter latitude and longitude manually.")

    lat = st.session_state.get("detected_lat")
    lon = st.session_state.get("detected_lon")
    if lat is None:
        lat = lat_input
    if lon is None:
        lon = lon_input

    if lat is not None and lon is not None:
        weather = fetch_weather(float(lat), float(lon))
        if weather and weather.get("temperature") is not None:
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("🌡️ Temperature", f"{weather['temperature']} °C")
            w2.metric("💧 Humidity", f"{weather['humidity']} %")
            w3.metric("🌧️ Precipitation", f"{weather['precipitation']} mm")
            w4.metric("💨 Wind", f"{weather['wind']} km/h")

    if not images:
        st.info("Upload a leaf photo above to start a diagnosis.")
        return

    if st.button("🔍 Diagnose", type="primary"):
        for image in images:
            top3 = predict_top3(model, image)
            top_class, top_conf = top3[0]
            info = DISEASE_INFO.get(top_class, {})
            record = {
                "time": datetime.now(),
                "filename": getattr(image, "name", "camera photo"),
                "image": image.getvalue(),
                "prediction": top_class,
                "confidence": top_conf,
                "top3": top3,
                "location": location_name,
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
            }
            add_diagnosis(record)

            img_col, res_col = st.columns([1, 2])
            with img_col:
                st.image(image, caption=record["filename"],
                         use_container_width=True)
            with res_col:
                low = top_conf < CONFIDENCE_THRESHOLD
                card_class = "result-card low" if low else "result-card"
                status = ("⚠️ Low confidence — treat as uncertain"
                          if low else "✅ Confident prediction")
                st.markdown(
                    f'<div class="{card_class}"><h3>{pretty_class(top_class)}'
                    f"</h3><p>{status} · {top_conf * 100:.1f}% confidence"
                    f"</p></div>",
                    unsafe_allow_html=True,
                )
                st.progress(min(max(top_conf, 0.0), 1.0))
                st.markdown("**Top 3 predictions**")
                st.dataframe(
                    pd.DataFrame(
                        [(pretty_class(c), f"{p * 100:.1f}%")
                         for c, p in top3],
                        columns=["Disease", "Confidence"],
                    ),
                    hide_index=True, use_container_width=True,
                )
                if low:
                    st.warning(
                        "The model's confidence is below 80%, so this result "
                        "is uncertain. Try a clearer, well-lit photo of a "
                        "single leaf, or check the other predictions above."
                    )
                if info:
                    with st.expander(f"About {pretty_class(top_class)}"):
                        st.markdown(f"**Symptoms:** {info['symptoms']}")
                        st.markdown(f"**What to do:** {info['management']}")
                        st.markdown(f"**Prevention:** {info['prevention']}")
        st.success("Diagnosis saved to History & Calendar.")

        points = [
            {"lat": r["lat"], "lon": r["lon"]}
            for r in st.session_state["history"]
            if r["lat"] is not None and r["lon"] is not None
        ]
        if points:
            st.markdown("### 🗺️ Diagnosis locations")
            st.map(pd.DataFrame(points), zoom=5)


def page_dashboard() -> None:
    """Dashboard page: session statistics and charts."""
    hero("📊 Dashboard",
         "An overview of this session's diagnoses and the model's verified "
         "performance.")
    st.markdown("### Model performance (verified, retrain #2, 7 Nov 2026)")
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, value) in zip((m1, m2, m3, m4), MODEL_STATS.items()):
        col.metric(label, value)
    st.caption("Validation accuracy peaked at 94.32% (epoch 9); macro F1 "
               "0.9417 on 17,572 validation images across 38 classes.")

    history = st.session_state["history"]
    st.markdown("### This session")
    if not history:
        st.info("No diagnoses yet in this session. Go to **Diagnose** and "
                "upload a leaf photo — the results will appear here.")
        return
    healthy = sum(1 for r in history
                  if DISEASE_INFO.get(r["prediction"], {}).get("status") == "healthy")
    diseased = len(history) - healthy
    avg_conf = float(np.mean([r["confidence"] for r in history])) * 100
    counts = pd.Series([pretty_class(r["prediction"]) for r in history]).value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total diagnoses", len(history))
    c2.metric("Healthy", healthy)
    c3.metric("Diseased", diseased)
    c4.metric("Average confidence", f"{avg_conf:.1f}%")
    st.markdown("### Diagnoses by disease")
    st.bar_chart(counts)
    df = history_frame()
    df["Date"] = pd.to_datetime(df["Date"])
    by_day = df.groupby("Date").size()
    st.markdown("### Diagnoses by day")
    st.bar_chart(by_day)
    st.markdown("### Recent diagnoses")
    st.dataframe(df.head(10), hide_index=True, use_container_width=True)

def page_history() -> None:
    """History & Calendar page: month calendar, seasonal guide, export."""
    hero("📅 History & Calendar",
         "Browse past diagnoses by day, check the seasonal disease guide, "
         "and export your history.")
    history = st.session_state["history"]
    if not history:
        st.info("No diagnoses yet in this session. Diagnoses you make on "
                "the **Diagnose** page will appear on this calendar.")
    else:
        st.markdown("### Diagnosis calendar")
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("◀ Previous month"):
                month = st.session_state["cal_month"] - 1
                year = st.session_state["cal_year"]
                if month == 0:
                    month, year = 12, year - 1
                st.session_state["cal_month"], st.session_state["cal_year"] = month, year
        with nav2:
            st.markdown(
                f"<h3 style='text-align:center'>{calmod.month_name[st.session_state['cal_month']]} "
                f"{st.session_state['cal_year']}</h3>",
                unsafe_allow_html=True,
            )
        with nav3:
            if st.button("Next month ▶"):
                month = st.session_state["cal_month"] + 1
                year = st.session_state["cal_year"]
                if month == 13:
                    month, year = 1, year + 1
                st.session_state["cal_month"], st.session_state["cal_year"] = month, year

        counts_by_day: dict = {}
        for rec in history:
            key = rec["time"].date()
            counts_by_day[key] = counts_by_day.get(key, 0) + 1
        year = st.session_state["cal_year"]
        month = st.session_state["cal_month"]
        header = st.columns(7)
        for col, name in zip(header, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            col.markdown(f"**{name}**")
        for week in calmod.monthcalendar(year, month):
            cols = st.columns(7)
            for col, day in zip(cols, week):
                if day == 0:
                    col.write("")
                    continue
                day_date = date(year, month, day)
                n = counts_by_day.get(day_date, 0)
                label = f"{day} ({n})" if n else str(day)
                if col.button(label, key=f"cal-{year}-{month}-{day}",
                              use_container_width=True):
                    st.session_state["cal_day"] = day_date

        selected = st.session_state["cal_day"]
        st.markdown(f"### Diagnoses on {selected.strftime('%d %B %Y')}")
        day_records = [r for r in history if r["time"].date() == selected]
        if not day_records:
            st.caption("No diagnoses on this day — pick a highlighted day.")
        for rec in day_records:
            with st.container():
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(rec["image"], use_container_width=True)
                with c2:
                    st.markdown(f"**{pretty_class(rec['prediction'])}** — "
                                f"{rec['confidence'] * 100:.1f}% confidence")
                    st.caption(
                        f"{rec['time'].strftime('%H:%M')} · "
                        f"{rec['filename']} · "
                        f"{rec['location'] or 'No location'}"
                    )

        st.markdown("### All history")
        st.dataframe(history_frame(), hide_index=True, use_container_width=True)
        csv = history_frame().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download history (CSV)", data=csv,
                           file_name="diagnosis_history.csv",
                           mime="text/csv")
        if st.button("🗑️ Clear history"):
            st.session_state["history"] = []
            st.rerun()

    st.markdown("### 🌦️ Seasonal disease guide")
    st.caption("General guidance for South Asia — typical patterns, not "
               "measured data. Always confirm with local conditions.")
    month_names = list(calmod.month_name)[1:]
    sel_month = st.selectbox("Select a month", month_names,
                             index=date.today().month - 1)
    month_no = month_names.index(sel_month) + 1
    st.info(SEASONAL_NOTES[month_no])
    st.markdown("**Typically active:** " + ", ".join(SEASONAL_ACTIVE[month_no]))


def page_library() -> None:
    """Disease Library page: searchable reference for all 38 classes."""
    hero("📚 Disease Library",
         "All 38 classes the model recognises, with symptoms and general "
         "management advice.")
    query = st.text_input("🔎 Search by crop or disease",
                          placeholder="e.g. tomato, blight, healthy")
    crops = sorted({info["crop"] for info in DISEASE_INFO.values()})
    crop_filter = st.selectbox("Filter by crop", ["All crops"] + crops)
    status_filter = st.selectbox("Show", ["All", "Diseased only", "Healthy only"])

    shown = 0
    for class_name in CLASS_NAMES:
        info = DISEASE_INFO[class_name]
        if crop_filter != "All crops" and info["crop"] != crop_filter:
            continue
        if status_filter == "Diseased only" and info["status"] != "diseased":
            continue
        if status_filter == "Healthy only" and info["status"] != "healthy":
            continue
        if query and query.lower() not in (
                info["crop"] + " " + info["label"] + " " + class_name).lower():
            continue
        shown += 1
        badge = "🟢 Healthy" if info["status"] == "healthy" else "🔴 Diseased"
        with st.expander(f"{pretty_class(class_name)} — {badge}"):
            st.markdown(f"**Symptoms:** {info['symptoms']}")
            st.markdown(f"**What to do:** {info['management']}")
            st.markdown(f"**Prevention:** {info['prevention']}")
    st.caption(f"Showing {shown} of {len(CLASS_NAMES)} classes.")
    st.info("This library gives general cultural guidance only. For chemical "
            "treatments, ask your local agriculture extension office for "
            "approved options in your area.")


def page_about() -> None:
    """About page: the project, the model and its verified results."""
    hero("ℹ️ About This Project",
         "Leaf Disease Prediction Using Deep Learning — Final Year Project.")
    st.markdown(
        """
        This system uses a convolutional neural network to detect plant
        diseases from images of leaves. It was trained on a dataset of
        about 87,000 images of healthy and diseased crop leaves, divided
        into **38 categories** (train: 70,295 images, validation: 17,572,
        test: 33 sample images).

        The project aims to help farmers and gardeners identify plant
        diseases quickly, so they can act early, protect their crop and
        improve yields.
        """
    )
    st.markdown("### Model (retrain #2, 7 Nov 2026)")
    st.markdown(
        """
        - **Architecture:** custom CNN (convolution blocks 32/64/128/256/512,
          dense 1500, dropout), trained 10 epochs on Google Colab (T4 GPU)
        - **Validation accuracy:** 94.21% (peak 94.32% at epoch 9)
        - **Macro F1:** 0.9417 on 17,572 validation images
        - **Input:** 128×128 RGB images, pixels normalised to [0, 1]
        - **Prediction rule:** the top prediction is shown with its
          confidence; results below 80% confidence are marked uncertain
        """
    )
    st.markdown("### Links")
    st.markdown(
        "- GitHub: [Asadshah-coder/leaf-disease-prediction]"
        "(https://github.com/Asadshah-coder/leaf-disease-prediction)\n"
        "- Live demo: [leaf-disease-prediction-app.streamlit.app]"
        "(https://leaf-disease-prediction-app.streamlit.app/)"
    )
    st.warning(
        "This app is a decision-support tool, not a substitute for expert "
        "diagnosis. For serious outbreaks, confirm with your local "
        "agriculture extension office."
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Plant Disease Recognition System",
        page_icon="🌱",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()

    model, model_path = load_model()

    with st.sidebar:
        st.title("🌱 Plant Disease AI")
        st.caption("Leaf Disease Prediction Using Deep Learning")
        page = st.radio("Navigate", PAGES,
                        index=PAGES.index(st.session_state["page"]),
                        label_visibility="collapsed")
        st.session_state["page"] = page
        st.divider()
        st.markdown("### Model status")
        if model is not None:
            st.success(f"✅ Model loaded: `{model_path.name}`")
        else:
            st.error("❌ No model file found")
        st.markdown("### At a glance")
        for label, value in MODEL_STATS.items():
            st.write(f"**{label}:** {value}")
        st.divider()
        st.caption(
            "History is saved for this session only on the free tier. "
            "Diagnoses: "
            f"{len(st.session_state['history'])}"
        )
        st.caption(
            "[GitHub](https://github.com/Asadshah-coder/leaf-disease-prediction)"
        )

    if page == PAGES[0]:
        page_diagnose(model)
    elif page == PAGES[1]:
        page_dashboard()
    elif page == PAGES[2]:
        page_history()
    elif page == PAGES[3]:
        page_library()
    else:
        page_about()


if __name__ == "__main__":
    main()
