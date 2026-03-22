#!/usr/bin/env python3
# Cooking Assistant — Platform Independent

import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from app.audio import find_alsa_device
from app.camera import Camera
from app.config import Config
from app.emotion import EmotionDetector
from app.llm import LLM
from app.movements import MovementController
from app.pipeline import (
    SAMPLE_RATE,
    MicRecorder,
    load_silero,
    stream_and_speak,
    vad_loop,
)
from app.reachy import connect as connect_reachy
from app.stt import STT
from app.tts import create_tts
from app.web import Broadcaster, start_web_server
from app.rag import KnowledgeBase, RAGRetriever

console = Console()


def handle_memory_storage(kb: KnowledgeBase, text: str):
    """Check if the user wants Reachy to remember something."""
    prefixes = ["remember that", "please remember", "keep in mind that", "save this:"]
    for p in prefixes:
        if text.lower().startswith(p):
            fact = text[len(p) :].strip()
            if fact:
                kb.add_document(fact, metadata={"type": "cooking_memory"})
                return f"Got it! I've added that to my memory: {fact}"
    return None


# ── Cooking Helpers ───────────────────────────────────────────


def buy_from_instacart(ingredients: list[str]):
    """Placeholder for Instacart integration."""
    console.print(
        f"[bold green]🛒 Instacart Placeholder:[/bold green] Ordering {', '.join(ingredients)}"
    )
    return True


def sign_with_antennas(mover: MovementController, sentence: str):
    """Placeholder for Sign Language support via antennas."""
    if mover:
        console.print(
            f"[bold blue]🤟 Sign Language Placeholder:[/bold blue] Signing for: {sentence}"
        )
        mover.perform_sign("placeholder")


def handle_timers(text: str):
    """Simple regex-based timer detection."""
    match = re.search(r"timer (?:for )?(\d+) (minute|second|hour)", text, re.IGNORECASE)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        secs = amount
        if "minute" in unit:
            secs *= 60
        elif "hour" in unit:
            secs *= 3600

        def _timer():
            console.print(
                f"[bold yellow]⏰ Timer started for {amount} {unit}[/bold yellow]"
            )
            time.sleep(secs)
            console.print(f"[bold red]🔔 TIMER UP: {amount} {unit}![/bold red]")
            # Reachy can say something here if we want

        threading.Thread(target=_timer, daemon=True).start()
        return f"I've set a timer for {amount} {unit}."
    return None


# ── Main ─────────────────────────────────────────────────────


def main():
    config = Config.load()
    broadcaster = Broadcaster()

    console.print(
        Panel.fit(
            "[bold green]Reachy Mini Cooking Assistant[/bold green]\n"
            "Platform Independent - Powered by Cloud APIs\n"
            "Interactive vision + Speech + Signs",
            border_style="green",
        )
    )

    # ── Web UI ──────────────────────────────────────────────────
    start_web_server(broadcaster, host=config.web.host, port=config.web.port)
    console.print(f"  ✓ Web UI: http://{config.web.host}:{config.web.port}")

    # ── Reachy Mini ──────────────────────────────────────────────
    reachy = connect_reachy(config, console)

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(
        name_hint=config.audio.input_device or "Reachy Mini Audio"
    )
    hw = f"hw:{result[0]},{result[1]}" if result else "default"
    if not result:
        console.print("[yellow]  ⚠ Mic not found, trying default[/yellow]")

    # ── Camera setup ─────────────────────────────────────────────
    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if cam.start():
        console.print("  ✓ Camera started")
    else:
        console.print("[red]  ✗ Camera failed[/red]")
        return

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading Models...[/bold]")

    stt = STT(
        backend=config.stt.backend,
        api_key=config.stt.api_key,
        model=config.stt.model,
        language=config.stt.language,
    )
    stt.load()
    console.print(f"  ✓ STT ({config.stt.backend})")

    silero_model = load_silero(console) if config.vad.use_silero else None

    llm = LLM(
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        backend=config.llm.backend,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        system_prompt=config.cooking.system_prompt,
    )
    llm.load()
    console.print(f"  ✓ LLM ({llm.model})")

    tts = create_tts(
        backend=config.tts.backend,
        api_key=config.tts.api_key,
        voice=config.tts.voice,
        speed=config.tts.speed,
    )
    if tts.load():
        console.print(f"  ✓ TTS ({config.tts.backend})")
    else:
        console.print("  ⚠ TTS unavailable")
        tts = None

    # ── Memory (RAG) ─────────────────────────────────────────────
    kb = KnowledgeBase(
        persist_dir=config.rag.persist_dir,
        embedding_backend=config.rag.embedding_backend,
        embedding_model=config.rag.embedding_model,
        embedding_base_url=config.rag.embedding_base_url,
        api_key=config.llm.api_key,  # Use same key for embeddings
    )
    if config.rag.enabled:
        n_chunks, rebuilt = kb.sync_directory(config.rag.knowledge_dir)
        status = "rebuilt" if rebuilt else "loaded"
        console.print(f"  ✓ Memory ({status}, {n_chunks} chunks)")

    retriever = RAGRetriever(
        kb, n_results=config.rag.n_results, min_relevance=config.rag.min_relevance
    )

    mover = (
        MovementController(reachy, config.reachy.antenna_rest_position)
        if reachy
        else None
    )
    emotion_detector = EmotionDetector() if config.emotion.enabled else None
    if emotion_detector:
        emotion_detector.load()

    # ── Mic Recorder ──────────────────────────────────────────────
    mic = MicRecorder(console, chunk_ms=32 if silero_model else config.vad.chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]✗ Mic failed[/red]")
        return

    console.print("\n[bold green]Ready to cook![/bold green] Just speak to Reachy.\n")

    # ── Proactive Vision Thread ───────────────────────────────────
    # We broadcast frames to the web UI frequently
    def _vision_broadcaster():
        while True:
            frame = cam.get_latest_frame()
            if frame:
                broadcaster.send({"type": "image", "data": frame})
            time.sleep(1.0 / config.web.ui_fps)

    threading.Thread(target=_vision_broadcaster, daemon=True).start()

    # ── Main Loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            # Capture what Reachy was seeing DURING the speech
            captured_frames = cam.get_speech_frames(
                speech_start=segment.start_time,
                speech_end=segment.end_time,
                max_frames=config.vision.frames,
            )

            # Transcribe
            console.print(f"  [cyan]●[/cyan] Transcribing {segment.duration:.1f}s audio...")
            result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
            text = result.get("text", "").strip()
            err = result.get("error", "")

            if not text:
                if err:
                    console.print(f"  [red]  (STT Error: {err})[/red]")
                else:
                    console.print("  [dim]  (STT: no text returned)[/dim]")
                mic.resume()
                continue

            console.print(f'  [green]You:[/green] "{text}"')
            broadcaster.send({"type": "chat", "role": "user", "text": text})

            # Memory storage check
            mem_response = handle_memory_storage(kb, text)
            if mem_response:
                console.print(f"  [magenta]Assistant:[/magenta] {mem_response}")
                broadcaster.send({"type": "chat", "role": "assistant", "text": mem_response})
                if tts: tts.synthesize_to_file(mem_response, "/tmp/resp.wav")
                mic.resume()
                continue

            # Agentic tasks check
            timer_response = handle_timers(text)
            if timer_response:
                console.print(f"  [magenta]Assistant:[/magenta] {timer_response}")
                broadcaster.send({"type": "chat", "role": "assistant", "text": timer_response})
                if tts: tts.synthesize_to_file(timer_response, "/tmp/resp.wav")
                mic.resume()
                continue

            # Sign Language trigger
            if config.cooking.sign_language_enabled and ("sign" in text.lower() or "show me" in text.lower()):
                sign_with_antennas(mover, text)

            # RAG Retrieval
            prompt_text = text
            if config.rag.enabled:
                prompt_text = retriever.augment_query(text)

            # LLM Stream
            n_imgs = len(captured_frames)
            console.print(f"  [cyan]●[/cyan] Calling {config.llm.model} (VLM with {n_imgs} frames)...")
            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm,
                tts,
                prompt_text,
                config.cooking.system_prompt,
                mic.pa_sink,
                images_b64=captured_frames if captured_frames else None,
            )
            console.print()
            broadcaster.send({"type": "chat", "role": "assistant", "text": full_resp})

            # Movement react
            if emotion_detector and mover:
                emo = emotion_detector.detect(full_resp)  # Detect assistant emotion
                mover.react(emo.emotion, emo.confidence)

            mic.resume()

    except KeyboardInterrupt:
        pass

    console.print("[yellow]Shutting down...[/yellow]")
    mic.stop()
    cam.close()
    if reachy:
        reachy.goto_sleep()


if __name__ == "__main__":
    main()
