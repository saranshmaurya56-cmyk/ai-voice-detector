import base64
import uuid
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from detector import VoiceDetector

app = FastAPI()

API_KEY = "hackathon-secret-key"

# Load model ONCE at startup
print("Loading model at startup...")
detector = VoiceDetector()
print("Model loaded!")

class AudioRequest(BaseModel):
    audio_base64: str


@app.get("/")
def home():
    return {"message": "AI Voice Detector API running"}


@app.post("/detect")
def detect_audio(data: AudioRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        audio_bytes = base64.b64decode(data.audio_base64)

        filename = f"temp_{uuid.uuid4()}.wav"
        with open(filename, "wb") as f:
            f.write(audio_bytes)

        label, confidence = detector.predict(filename)

        os.remove(filename)

        return {
            "result": label,
            "confidence": round(float(confidence), 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
