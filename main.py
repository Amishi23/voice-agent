
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from stt import transcribe_audio
from llm import get_reply
from tts import text_to_speech_base64
from memory import init_db, save_message, get_history
from classifier import classify_intent

app = FastAPI()
init_db()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    with open("frontend/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[+] Session connected: {session_id}")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"[>] Received {len(data)} bytes of audio")

            await websocket.send_json({"type": "status", "text": "Transcribing..."})
            transcript = transcribe_audio(data)
            print(f"[STT] {transcript}")

            if not transcript:
                await websocket.send_json({"type": "error", "text": "Could not transcribe audio"})
                continue

            intent_result = classify_intent(transcript)
            print(f"[INTENT] {intent_result['intent']} ({intent_result['confidence']}%)")

            save_message(session_id, "user", transcript)
            await websocket.send_json({
                "type": "transcript",
                "text": transcript,
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"]
            })

            await websocket.send_json({"type": "status", "text": "Thinking..."})
            history = get_history(session_id)
            reply = get_reply(history)
            print(f"[LLM] {reply}")

            save_message(session_id, "assistant", reply)

            await websocket.send_json({"type": "status", "text": "Speaking..."})
            audio_b64 = text_to_speech_base64(reply)

            await websocket.send_json({
                "type": "reply",
                "text": reply,
                "audio": audio_b64
            })

            await websocket.send_json({"type": "status", "text": "Ready"})

    except WebSocketDisconnect:
        print(f"[-] Session disconnected: {session_id}")