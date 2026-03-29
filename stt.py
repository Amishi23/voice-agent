import whisper
import tempfile
import os
import imageio_ffmpeg

# ✅ Fix FFmpeg path (required for Whisper)
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

# ✅ Load model once (important)
# You can change "tiny" → "base" for better accuracy
model = whisper.load_model("tiny", download_root="./models")


def transcribe_audio(audio_bytes: bytes) -> str:
    tmp_path = None

    try:
        # ✅ Save incoming audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # ✅ Faster + stable transcription settings
        result = model.transcribe(
            tmp_path,
            fp16=False,          # CPU safe
            language="en",       # force English (faster)
            task="transcribe",   # no translation
            temperature=0.0,     # deterministic
            best_of=1,           # faster
            beam_size=1          # faster
        )

        text = result.get("text", "").strip()
        return text

    except Exception as e:
        print("[STT ERROR]", e)
        return ""

    finally:
        # ✅ Always delete temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_error:
                print("[STT CLEANUP ERROR]", cleanup_error)