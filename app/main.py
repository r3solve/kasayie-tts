from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
import uuid
from pydub import AudioSegment
import io
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from gtts import gTTS

# Initialize FastAPI app
app = FastAPI(title="Kasayie TTS API")

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
SUPABASE_URL = "https://rcgkpwiuviamuihrrfuw.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJjZ2twd2l1dmlhbXVpaHJyZnV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA2NjI0NjAsImV4cCI6MjA1NjIzODQ2MH0.BzU2TOnzB1dY-mbk6mrkiknk_mN-IGvYePbBEfvuiiQ"    # Replace with your Supabase anon or service key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize TTS pipeline from Huggingface transformers
pipe = pipeline("text-to-audio", model="d3vnerd/TTS_twi_test")

# Pydantic model for request body
class TTSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"detail": "Welcome to Kasayie TTS. Visit /docs for documentation"}

@app.post("/generate_tts/")
async def generate_tts(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    try:
        # Generate audio from text
        output = pipe(text)
        audio = output["audio"]
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # Normalize and convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        # Convert float32 numpy array to int16 for pydub
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create AudioSegment from raw audio data
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=output["sampling_rate"],
            sample_width=2,  # 2 bytes for int16
            channels=1       # assuming mono audio
        )

        # Export audio to in-memory bytes buffer as MP3
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"

        # Upload to Supabase storage bucket named "audio"
        response = supabase.storage.from_('audio').upload(filename, mp3_buffer, {'content-type': 'audio/mpeg'})

        if response.get('error'):
            raise HTTPException(status_code=500, detail=f"Supabase upload error: {response['error']['message']}")

        # Get public URL for uploaded file
        public_url = supabase.storage.from_('audio').get_public_url(filename)

        return {"audio_url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.post("/generate_eng_tts/")
async def generate_eng_tts(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    try:
        # Generate TTS using gTTS
        tts = gTTS(text=text, lang='en')

        # Save to in-memory bytes buffer
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"

        # Upload to Supabase storage bucket "audio"
        response = supabase.storage.from_('audio').upload(filename, mp3_buffer, {'content-type': 'audio/mpeg'})

        if response.get('error'):
            raise HTTPException(status_code=500, detail=f"Supabase upload error: {response['error']['message']}")

        # Get public URL for uploaded file
        public_url = supabase.storage.from_('audio').get_public_url(filename)

        return {"audio_url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS processing error: {str(e)}")
