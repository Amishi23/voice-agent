with open("test_audio.webm", "rb") as f:
    audio_bytes = f.read()

from stt import transcribe_audio
print(transcribe_audio(audio_bytes))