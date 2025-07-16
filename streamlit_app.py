import streamlit as st
from deepface.DeepFace import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import whisper
from transformers import pipeline
import json
import datetime
import os

st.set_page_config(page_title="ParentPulse", layout="centered")

# Load models only once
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    sentiment_model = pipeline("sentiment-analysis")
    return whisper_model, sentiment_model

whisper_model, sentiment_model = load_models()

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

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.emotion = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            self.emotion = result[0]['dominant_emotion']
        except Exception:
            self.emotion = None
        return img

webrtc_ctx = webrtc_streamer(
    key="emotion-capture",
    video_transformer_factory=EmotionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

face_emotion = None
if webrtc_ctx.video_transformer:
    face_emotion = webrtc_ctx.video_transformer.emotion

if face_emotion:
    st.success(f"Detected facial emotion: {face_emotion}")

st.subheader("Optional: Upload a voice recording")

audio_file = st.file_uploader("Upload voice clip (wav/mp3)", type=["wav", "mp3"])
voice_sentiment = None
transcript = ""

if audio_file:
    with st.spinner("Transcribing voice..."):
        result = whisper_model.transcribe(audio_file)
        transcript = result["text"]
        st.write(f"Transcription: {transcript}")
        voice_sentiment = sentiment_model(transcript)[0]
        st.success(f"Voice sentiment: {voice_sentiment['label']} ({voice_sentiment['score']:.2f})")

if st.button("✅ Submit Check-in"):
    if not face_emotion and not transcript:
        st.warning("Please provide either webcam or voice input.")
    else:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "face_emotion": face_emotion,
            "voice_transcript": transcript,
            "voice_sentiment": voice_sentiment,
        }
        save_log(entry)
        st.success("Check-in logged!")

        # Notification logic
        if (face_emotion and face_emotion in ["angry", "sad", "fear", "disgust"]) or \
           (voice_sentiment and voice_sentiment["label"] == "NEGATIVE"):
            st.error("⚠️ Negative emotion detected! Consider calling your parent.")

st.markdown("---")
st.subheader("📋 Check-in History")

logs = load_logs()
if logs:
    for log in reversed(logs[-10:]):
        st.write(f"🕒 {log['timestamp']}")
        if log["face_emotion"]:
            st.write(f"😊 Facial: {log['face_emotion']}")
        if log.get("voice_sentiment"):
            st.write(f"🗣️ Voice: {log['voice_sentiment']['label']} ({log['voice_sentiment']['score']:.2f})")
        if log.get("voice_transcript"):
            st.caption(f"📝 {log['voice_transcript']}")
        st.markdown("---")
else:
    st.info("No logs yet.")
