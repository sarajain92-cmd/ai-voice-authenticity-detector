import streamlit as st
import requests

st.set_page_config(page_title="Voice Deepfake Detector", page_icon="🎙️")

st.title("🎙️ Voice Deepfake Detector")
st.markdown("Upload a `.wav` audio file to check if it's **REAL** or **FAKE**")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing audio..."):
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": uploaded_file.getvalue()}
            )
            result = response.json()
        
        label = result["label"]
        confidence = result["confidence"]
        
        if label == "FAKE":
            st.error(f"⚠️ DEEPFAKE DETECTED — Confidence: {confidence*100:.1f}%")
        else:
            st.success(f"✅ REAL VOICE — Confidence: {confidence*100:.1f}%")