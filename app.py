# app.py

import re
import pickle
import pathlib
import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Sentiment Analyzer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    base_path = pathlib.Path(__file__).parent
    model_path = base_path / "model.pkl"
    vectorizer_path = base_path / "vectorizer.pkl"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' exist in the app directory."
        )

    with open(model_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)

    with open(vectorizer_path, "rb") as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    return loaded_model, loaded_vectorizer

try:
    model, vectorizer = load_model()
except Exception as err:
    st.error("Unable to load the sentiment model.")
    st.error(str(err))
    st.stop()

# ---------------- STYLES ----------------
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #fff7ed 0%, #ffe8d6 30%, #ffd8a8 60%, #f8b35e 100%);
            color: #1f2937;
        }

        .block-container {
            max-width: 1240px;
            padding: 3.8rem 2.2rem 3rem;
        }

        .hero-card {
            position: relative;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(251, 146, 60, 0.16);
            border-radius: 40px;
            padding: 3rem 2.5rem;
            box-shadow: 0 35px 110px rgba(249, 115, 22, 0.16);
            overflow: hidden;
            margin-bottom: 1.8rem;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            background: rgba(251, 146, 60, 0.16);
            border-radius: 50%;
            top: -50px;
            right: -50px;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 120px;
            height: 120px;
            background: rgba(251, 146, 60, 0.08);
            border-radius: 50%;
            bottom: -40px;
            left: -40px;
        }

        .hero-title {
            font-size: clamp(3rem, 5vw, 4.6rem);
            font-weight: 900;
            margin-bottom: 1rem;
            color: #111827;
        }

        .hero-subtitle {
            font-size: 1.18rem;
            color: #334155;
            max-width: 780px;
            line-height: 1.9;
            margin-bottom: 0;
            font-weight: 500;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.9rem 1.2rem;
            background: rgba(251, 146, 60, 0.16);
            color: #b45309;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .control-card {
            background: rgba(255, 255, 255, 0.97);
            border: 1px solid rgba(251, 146, 60, 0.22);
            border-radius: 32px;
            padding: 2.4rem;
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.08);
            margin-bottom: 2rem;
        }

        .form-section {
            display: flex;
            gap: 2rem;
            align-items: flex-start;
        }

        .form-section .stTextArea textarea {
            min-height: 260px !important;
        }

        .form-section .stFileUploader {
            width: 100% !important;
        }

        .stTextArea textarea,
        .stTextInput>div>div>input {
            border-radius: 22px !important;
            border: 1px solid rgba(251, 146, 60, 0.24) !important;
            background-color: #fff7f0 !important;
            color: #111827 !important;
            padding: 1.1rem !important;
            font-size: 1rem !important;
            box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.06);
        }

        .stRadio > div {
            border-radius: 20px;
            padding: 0.7rem 0.95rem;
            background: rgba(251, 146, 60, 0.08);
        }

        .stRadio input[type="radio"] + label {
            font-weight: 700;
            font-size: 1rem;
            color: #111827;
        }

        .stButton > button {
            background: #ea580c;
            color: white;
            border-radius: 18px;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.05rem;
            font-weight: 700;
            box-shadow: 0 16px 36px rgba(234, 88, 12, 0.22);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 44px rgba(234, 88, 12, 0.28);
        }

        .stButton > button:focus {
            outline: none;
            box-shadow: 0 0 0 4px rgba(251, 146, 60, 0.18);
        }

        .info-card {
            background: #fff8f1;
            border: 1px solid rgba(251, 146, 60, 0.18);
            border-radius: 24px;
            padding: 1.6rem;
        }

        .info-card h3 {
            margin-top: 0;
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
        }

        .info-card ul {
            padding-left: 1.3rem;
            margin-top: 0.9rem;
            color: #334155;
            font-size: 1rem;
        }

        .info-card li {
            margin-bottom: 0.85rem;
            line-height: 1.9;
        }

        .result-box {
            border-radius: 24px;
            padding: 1.6rem;
            background: linear-gradient(180deg, rgba(251, 146, 60, 0.08), rgba(255, 255, 255, 0.96));
            border: 1px solid rgba(251, 146, 60, 0.22);
            margin-top: 1.4rem;
        }

        .footer-note {
            text-align: center;
            color: #334155;
            margin-top: 2.2rem;
            margin-bottom: 0.6rem;
            font-size: 0.98rem;
        }

        @media (max-width: 950px) {
            .block-container {
                padding: 2.4rem 1.2rem 2.2rem;
            }

            .form-section {
                flex-direction: column;
            }

            .hero-card {
                padding: 2.2rem 1.6rem;
            }
        }

        @media (max-width: 720px) {
            .hero-card::before,
            .hero-card::after {
                display: none;
            }

            .hero-title {
                font-size: 2.6rem;
            }

            .hero-subtitle {
                font-size: 1rem;
            }

            .stTextArea textarea {
                min-height: 220px !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
with st.container():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">🍔 Premium food sentiment insights</div>
            <h1 class="hero-title">Food Review Analyzer</h1>
            <p class="hero-subtitle">
                A clean, modern food review website experience for restaurants and menu teams. Analyze sentiment quickly, compare positive and negative feedback, and export your insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- CONTROLS ----------------
mode = st.radio("Choose prediction mode", ["Single Review", "Bulk Reviews"], horizontal=True)

st.markdown("<div class='control-card'>", unsafe_allow_html=True)
with st.form(key="sentiment_form"):
    if mode == "Single Review":
        left_col, right_col = st.columns([2.4, 1])

        with left_col:
            text_input = st.text_area(
                "✍️ Write a review",
                height=240,
                placeholder="The burger was juicy and flavorful, but the fries were too salty..."
            )

        with right_col:
            st.markdown(
                "<div class='info-card'><h3>Tips for best results</h3>"
                "<ul>"
                "<li>Keep each review short and specific.</li>"
                "<li>Enter one review at a time for accurate results.</li>"
                "<li>Use bulk CSV upload for many reviews.</li>"
                "</ul></div>",
                unsafe_allow_html=True,
            )
        file_upload = None
    else:
        text_input = ""
        st.markdown("## 📄 Bulk review upload")
        st.markdown("Upload a CSV file containing one review per row.")
        st.markdown("Supported column names: **review**, **text**, **comments**, **comment**.")
        file_upload = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Valid columns: review, text, comments, comment"
        )
        st.markdown(
            "<div class='info-card'><h3>Bulk upload tips</h3>"
            "<ul>"
            "<li>Upload only one review per row.</li>"
            "<li>Use the same column name throughout the file.</li>"
            "<li>Large uploads are handled smoothly.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button("🔍 Analyze Sentiment")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------

def parse_bulk_reviews(uploaded_file) -> list[str]:
    reviews = []

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            review_columns = [
                c for c in data.columns if c.lower() in ["review", "text", "comments", "comment"]
            ]
            if review_columns:
                reviews.extend(data[review_columns[0]].astype(str).tolist())
            else:
                reviews.extend(data.iloc[:, 0].astype(str).tolist())
        except Exception as exc:
            st.error(f"Unable to read uploaded file: {exc}")
            return []

    return [review for review in reviews if review.strip()]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.[^\s]+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_reviews(reviews: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cleaned_reviews = [clean_text(review) for review in reviews]
    review_vec = vectorizer.transform(cleaned_reviews)
    predictions = model.predict(review_vec)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(review_vec)
        confidences = np.max(proba, axis=1) * 100
    else:
        confidences = np.full(len(reviews), np.nan)

    return predictions, confidences

# ---------------- PREDICTION ----------------
if submitted:
    if mode == "Single Review":
        if not text_input.strip():
            st.warning("⚠️ Please enter a review before analysis.")
        else:
            prediction, confidence = predict_reviews([text_input])
            label = prediction[0]
            score = confidence[0]

            if label == "positive":
                st.success(f"😊 Positive review — {score:.2f}% confidence")
            elif label == "negative":
                st.error(f"😢 Negative review — {score:.2f}% confidence")
            else:
                st.info(f"😐 Neutral review — {score:.2f}% confidence")

            st.markdown(
                "<div class='result-box'><strong>Review text:</strong><br>"
                f"{text_input}" 
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        reviews = parse_bulk_reviews(file_upload)
        if not reviews:
            st.warning("⚠️ Please upload a valid CSV file with your bulk reviews.")
        else:
            labels, confidences = predict_reviews(reviews)
            results = pd.DataFrame(
                {
                    "Review": reviews,
                    "Sentiment": labels,
                    "Confidence (%)": np.round(confidences, 2),
                }
            )

            counts = results["Sentiment"].value_counts().reindex(
                ["positive", "neutral", "negative"], fill_value=0
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", int(counts["positive"]))
            col2.metric("Neutral", int(counts["neutral"]))
            col3.metric("Negative", int(counts["negative"]))

            st.markdown("### Bulk prediction results")
            st.dataframe(results, use_container_width=True)

            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download predictions as CSV",
                csv_data,
                file_name="food_sentiment_predictions.csv",
                mime="text/csv",
            )

            st.success("Bulk sentiment analysis completed successfully.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569;'>Made with ❤️ using Streamlit</div>",
    unsafe_allow_html=True,
)
