"""TTS — pluggable text-to-speech backends (Piper and Kokoro)."""

import sys
import wave
from typing import Dict, Any
from pathlib import Path
import numpy as np


VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

# Kokoro v1.0 model files (auto-downloaded if missing)
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_kokoro_models_if_missing() -> bool:
    """Download Kokoro model and voices to voices/ if not present. Returns True if both files exist (after optional download)."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = VOICES_DIR / "kokoro-v1.0.onnx"
    voices_path = VOICES_DIR / "voices-v1.0.bin"
    needed = []
    if not model_path.exists():
        needed.append((KOKORO_MODEL_URL, model_path, "kokoro-v1.0.onnx (~311 MB)"))
    if not voices_path.exists():
        needed.append((KOKORO_VOICES_URL, voices_path, "voices-v1.0.bin (~30 MB)"))
    if not needed:
        return True
    try:
        import httpx
    except ImportError:
        print("Kokoro: install httpx to auto-download models (pip install httpx)")
        return False
    for url, path, label in needed:
        print(f"Downloading {label} to {path} ...")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                done = 0
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=262144):
                        f.write(chunk)
                        done += len(chunk)
                        if total and total > 0:
                            pct = 100 * done / total
                            sys.stdout.write(f"\r  {label}: {pct:.0f}%\r")
                            sys.stdout.flush()
            if total:
                print()
            print(f"  Saved {path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return False
    return True


class PiperTTS:
    """Piper neural TTS (CPU, lightweight, ~61 MB model)."""

    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._piper = None
        self._sample_rate = 22050
        self.backend_name = "Piper"

    def load(self) -> bool:
        try:
            from piper import PiperVoice

            search = [
                VOICES_DIR / f"{self.voice}.onnx",
                Path(self.voice),
                Path(f"{self.voice}.onnx"),
                Path.home() / ".local" / "share" / "piper" / "voices" / f"{self.voice}.onnx",
            ]
            for p in search:
                try:
                    if p.exists():
                        cfg = p.with_suffix(".json")
                        self._piper = PiperVoice.load(str(p), config_path=str(cfg) if cfg.exists() else None)
                        if hasattr(self._piper, "config") and self._piper.config:
                            self._sample_rate = getattr(self._piper.config, "sample_rate", 22050)
                        return True
                except Exception:
                    continue
            print(f"TTS voice not found: {self.voice}")
            return False
        except ImportError:
            print("piper-tts not installed")
            return False

    def synthesize(self, text: str) -> Dict[str, Any]:
        if self._piper is None:
            return {"audio": None, "error": "Not loaded"}
        if not text.strip():
            return {"audio": None, "error": "Empty"}
        try:
            chunks = []
            for chunk in self._piper.synthesize(text):
                chunks.append((chunk.audio_float_array * 32767).astype(np.int16))
            if not chunks:
                return {"audio": None, "error": "No audio"}
            return {"audio": np.concatenate(chunks), "sample_rate": self._sample_rate}
        except Exception as e:
            return {"audio": None, "error": str(e)}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        return self._piper is not None

    def unload(self):
        if self._piper:
            del self._piper
            self._piper = None


class KokoroTTS:
    """Kokoro neural TTS (ONNX Runtime, high quality, ~300 MB model)."""

    def __init__(self, voice: str = "af_sarah", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._kokoro = None
        self._sample_rate = 24000
        self.backend_name = "Kokoro"

    def load(self) -> bool:
        try:
            import os
            import onnxruntime as ort

            model_path = VOICES_DIR / "kokoro-v1.0.onnx"
            voices_path = VOICES_DIR / "voices-v1.0.bin"

            if not model_path.exists() or not voices_path.exists():
                if not _download_kokoro_models_if_missing():
                    return False
                model_path = VOICES_DIR / "kokoro-v1.0.onnx"
                voices_path = VOICES_DIR / "voices-v1.0.bin"
            if not model_path.exists() or not voices_path.exists():
                return False

            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
            elif "TensorrtExecutionProvider" in available:
                os.environ["ONNX_PROVIDER"] = "TensorrtExecutionProvider"

            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(model_path), str(voices_path))

            provider_used = self._kokoro.sess.get_providers()[0]
            print(f"Kokoro TTS loaded — ONNX provider: {provider_used}")
            return True
        except ImportError:
            print("kokoro-onnx not installed (pip install kokoro-onnx)")
            return False
        except Exception as e:
            print(f"Kokoro load error: {e}")
            return False

    def synthesize(self, text: str) -> Dict[str, Any]:
        if self._kokoro is None:
            return {"audio": None, "error": "Not loaded"}
        if not text.strip():
            return {"audio": None, "error": "Empty"}
        try:
            samples, sample_rate = self._kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.lang,
            )
            audio_int16 = (samples * 32767).astype(np.int16)
            return {"audio": audio_int16, "sample_rate": sample_rate}
        except Exception as e:
            return {"audio": None, "error": str(e)}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        return self._kokoro is not None

    def unload(self):
        if self._kokoro:
            del self._kokoro
            self._kokoro = None


# Keep backward compat: TTS = PiperTTS
TTS = PiperTTS


def create_tts(backend: str = "piper", voice: str = "", speed: float = 1.0,
               piper_voice: str = "en_US-lessac-medium", lang: str = "en-us"):
    """Factory: create the right TTS backend. Falls back to Piper if Kokoro model is missing."""
    if backend == "kokoro":
        k = KokoroTTS(voice=voice or "af_sarah", speed=speed, lang=lang)
        if k.load():
            return k
        print(
            "Kokoro model not found (download kokoro-v1.0.onnx and voices-v1.0.bin to voices/). "
            "Falling back to Piper."
        )
        p = PiperTTS(voice=piper_voice, speed=speed)
        p.load()
        return p
    return PiperTTS(voice=voice or piper_voice, speed=speed)
