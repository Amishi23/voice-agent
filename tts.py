from gtts import gTTS
import io
import base64

def text_to_speech_base64(text: str) -> str:
    tts = gTTS(text=text, lang="en", slow=False)

    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)

    audio_bytes = audio_buffer.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return audio_base64