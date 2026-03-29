# Voice AI Support Agent

A real-time voice AI agent built with Python, FastAPI, and LLaMA 3.3 70B.

## Features
- Press and hold mic to speak — agent replies with voice
- Whisper STT for speech-to-text transcription
- LLaMA 3.3 70B (Groq) for AI responses
- gTTS for text-to-speech voice output
- scikit-learn intent classifier (book/cancel/escalate) at 90%+ accuracy
- SQLite conversation memory across turns
- Live analytics dashboard with confidence scores

## Tech Stack
Python, FastAPI, WebSocket, Whisper, Groq API, scikit-learn, SQLite, gTTS

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
uvicorn main:app --reload
```
Open http://localhost:8000