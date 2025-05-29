from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import soundfile as sf
import numpy as np
import os
import uuid
from scipy.io.wavfile import write as scipy_write
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Kasayie TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = pipeline("text-to-audio", model="d3vnerd/TTS_twi_test")

AUDIO_DIR = "/static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

BASE_URL = "http://localhost:8000"

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"detail": "Welcome to Kasayie TTS. Visit /docs for documentation"}

from pydub import AudioSegment
import io

@app.post("/generate_tts/")
async def generate_tts(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    try:
        output = pipe(text)
        audio = output["audio"]
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # Normalize and convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        # Convert float32 numpy array (-1.0 to 1.0) to int16 for pydub
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create AudioSegment from raw audio data
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=output["sampling_rate"],
            sample_width=2,  # 2 bytes for int16
            channels=1       # assuming mono audio
        )

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Export as MP3
        audio_segment.export(filepath, format="mp3")

        audio_url = f"static/audio/{filename}"
        return {"audio_url": audio_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")


from gtts import gTTS

@app.post("/generate_eng_tts/")
async def generate_eng_tts(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    try:
        tts = gTTS(text=text, lang='en')
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Save the TTS output as an MP3 file
        tts.save(filepath)

        audio_url = f"static/audio/{filename}"
        return {"audio_url": audio_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS processing error: {str(e)}")


from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
