import streamlit as st
from PIL import Image
from fer import FER
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import whisper
from transformers import pipeline
import json
import datetime
import os
import numpy as np
import av

st.set_page_config(page_title="ParentPulse Remote Check-in", layout="centered")

# Load models once
@st.cache_resource
def load_models():
    face_detector = FER(mtcnn=True)
    whisper_model = whisper.load_model("base")
    sentiment_model = pipeline("sentiment-analysis")
    return face_detector, whisper_model, sentiment_model

face_detector, whisper_model, sentiment_model = load_models()

LOG_FILE = "checkin_logs.json"

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_log(entry):
    logs = load_logs()
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

st.title("👨‍👩‍👧 ParentPulse - Daily Emotion Check-in")

# Webcam capture class
class VideoEmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_emotion = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = face_detector.top_emotion(img)
        if result:
            emotion, score = result
            self.face_emotion = emotion
        return img

webrtc_ctx = webrtc_streamer(
    key="emotion-capture",
    video_transformer_factory=VideoEmotionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

face_emotion = None
if webrtc_ctx.video_transformer:
    face_emotion = webrtc_ctx.video_transformer.face_emotion

if face_emotion:
    st.success(f"Detected facial emotion: {face_emotion}")

st.subheader("Optional: Upload a voice clip (wav/mp3)")

audio_file = st.file_uploader("Upload voice recording", type=["wav", "mp3"])
voice_sentiment = None
transcript = ""

if audio_file:
    with st.spinner("Transcribing and analyzing voice..."):
        result = whisper_model.transcribe(audio_file)
        transcript = result["text"]
        st.write(f"Transcription: {transcript}")
        voice_sentiment = sentiment_model(transcript)[0]
        st.success(f"Voice sentiment: {voice_sentiment['label']} (Confidence: {voice_sentiment['score']:.2f})")

if st.button("Submit Check-in"):
    if not face_emotion and not transcript:
        st.error("Please provide either a webcam emotion or voice clip.")
    else:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "face_emotion": face_emotion,
            "voice_transcript": transcript,
            "voice_sentiment": voice_sentiment,
        }
        save_log(entry)
        st.success("Check-in saved!")

        # Alert if negative mood detected
        negative_emotions = ["sad", "angry", "disgust"]
        if (face_emotion and face_emotion.lower() in negative_emotions) or \
           (voice_sentiment and voice_sentiment["label"] == "NEGATIVE"):
            st.error("🚨 Negative emotion detected! Please check in with your parent.")

st.markdown("---")
st.subheader("Recent Check-ins")
logs = load_logs()
if logs:
    for log in reversed(logs[-10:]):
        st.write(f"🕒 {log['timestamp']}")
        if log["face_emotion"]:
            st.write(f"😊 Facial emotion: {log['face_emotion']}")
        if log.get("voice_sentiment"):
            st.write(f"🗣️ Voice sentiment: {log['voice_sentiment']['label']} (Confidence: {log['voice_sentiment']['score']:.2f})")
        if log.get("voice_transcript"):
            st.caption(f"📝 \"{log['voice_transcript']}\"")
        st.markdown("---")
else:
    st.info("No check-ins yet.")