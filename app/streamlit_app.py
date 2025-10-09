import streamlit as st
import joblib

st.set_page_config(page_title="Toxicity Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Toxicity Detector — Multiclass Demo")
st.caption("Classifies text as hate / offensive / neutral using a TF-IDF + Linear SVM baseline.")

@st.cache_resource
def load_model(path: str = "models/multiclass_tfidf_svm.joblib"):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle.get("label_map", {1: "hate", 2: "offensive", 3: "neutral"})

try:
    pipe, label_map = load_model()
except Exception as e:
    st.error(f"Could not load model. Ensure it exists at models/multiclass_tfidf_svm.joblib. Error: {e}")
    st.stop()

with st.form("predict_form"):
    text = st.text_area("Enter a sentence or short paragraph:", height=140, placeholder="Type here…")
    submitted = st.form_submit_button("Classify")

if submitted and text:
    pred = pipe.predict([text])[0]
    label = label_map.get(int(pred), str(pred))
    st.markdown(f"### Prediction: **{label}**")
    st.info("This is a baseline demonstration. For production, consider thresholding, bias checks, and an ensemble.")

st.divider()
st.subheader("Batch classify (optional)")
texts = st.text_area("One text per line:", height=120)
if st.button("Batch classify") and texts.strip():
    lines = [ln.strip() for ln in texts.splitlines() if ln.strip()]
    preds = pipe.predict(lines)
    out = [label_map.get(int(p), str(p)) for p in preds]
    for i, (t, lab) in enumerate(zip(lines, out), start=1):
        st.write(f"**{i}.** {lab} — {t[:120]}{'…' if len(t)>120 else ''}")

