#!/usr/bin/env python3
"""
Vision Chat — speak + see, the VLM describes what it sees.
Mic -> Silero/energy VAD -> [camera capture] -> STT -> VLM (text + images) -> TTS -> Speaker

Usage:
  python3 run_vision_chat.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.camera import Camera
from app.pipeline import (
    SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop, stream_and_speak, load_silero,
)
from rich.console import Console
from rich.panel import Panel

try:
    from reachy_mini import ReachyMini
    import psutil
    HAS_REACHY = True
except ImportError:
    HAS_REACHY = False
    psutil = None

console = Console()


def _is_reachy_daemon_running() -> bool:
    if not psutil:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            for part in cmdline:
                if "reachy-mini-daemon" in part:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return False


def _kill_reachy_daemon() -> bool:
    if not psutil:
        return False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            for part in cmdline:
                if "reachy-mini-daemon" in part:
                    pid = proc.pid
                    console.print(f"  [yellow]Killing stale Reachy daemon (PID {pid})[/yellow]")
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(2)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return False


def _kill_stale_camera_holders(device: int = 0):
    try:
        r = subprocess.run(
            ["fuser", f"/dev/video{device}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = r.stdout.strip().split()
        my_pid = str(os.getpid())
        for pid in pids:
            pid = pid.strip().rstrip("m")
            if pid and pid != my_pid:
                console.print(f"  [yellow]Killing stale process {pid} holding /dev/video{device}[/yellow]")
                os.kill(int(pid), signal.SIGKILL)
        if pids:
            time.sleep(0.5)
    except Exception:
        pass


def _connect_reachy(config):
    """Connect to Reachy Mini using config.reachy settings. Returns ReachyMini or None."""
    if not HAS_REACHY or not config.reachy.enabled:
        return None

    rcfg = config.reachy
    daemon_already_running = _is_reachy_daemon_running()

    for attempt in range(rcfg.daemon_retry_attempts):
        try:
            if attempt == 0:
                console.print("  Connecting to Reachy Mini...")
            elif attempt == 1:
                console.print(f"  [dim]Daemon may still be starting, waiting {rcfg.daemon_startup_wait:.0f}s...[/dim]")
                time.sleep(rcfg.daemon_startup_wait)
                console.print("  Retrying connection to Reachy Mini...")
            else:
                _kill_reachy_daemon()
                console.print("  Retrying connection (fresh daemon)...")

            reachy = ReachyMini(
                spawn_daemon=rcfg.spawn_daemon,
                use_sim=False,
                timeout=rcfg.timeout,
                media_backend=rcfg.media_backend,
            )

            reachy.enable_motors()
            if rcfg.wake_on_start:
                if daemon_already_running:
                    console.print("  Ensuring Reachy Mini is awake...")
                else:
                    console.print("  Waking up Reachy Mini...")
                reachy.wake_up()
                time.sleep(0.5)
                try:
                    reachy.set_target_antenna_joint_positions(rcfg.antenna_rest_position)
                    time.sleep(0.2)
                except Exception:
                    pass
                console.print("  [green]✓ Reachy Mini awake (head up, camera ready)[/green]")
            else:
                console.print("  [green]✓ Reachy Mini connected (wake_on_start=false)[/green]")
            return reachy

        except Exception as e:
            err_msg = str(e)
            if ("localhost and network" in err_msg or "both localhost" in err_msg.lower()) and attempt < rcfg.daemon_retry_attempts - 1:
                continue
            console.print(f"  [yellow]⚠ Reachy Mini unavailable: {e}[/yellow]")
            console.print("  [yellow]  Continuing without robot control[/yellow]")
            return None
    return None


def main():
    config = Config.load()

    console.print(Panel.fit(
        "[bold cyan]Vision Chat[/bold cyan]\n"
        "Speak anytime — camera captures when you speak\n"
        "[dim]Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Reachy Mini ──────────────────────────────────────────────
    reachy = _connect_reachy(config)

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini Audio")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Camera setup (background ring buffer) ────────────────────
    _kill_stale_camera_holders(config.vision.camera_device)

    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if cam.start():
        console.print(
            f"  ✓ Camera /dev/video{config.vision.camera_device} "
            f"({config.vision.width}x{config.vision.height}, "
            f"{config.vision.capture_fps} fps ring buffer)"
        )
    else:
        console.print("[red]  ✗ Camera not found! Check USB webcam.[/red]")
        return

    # ── Register cleanup for exit signals ────────────────────────
    def _cleanup(signum=None, frame=None):
        console.print("\n[yellow]Exiting...[/yellow]")
        cam.close()
        mic.stop()
        if reachy and config.reachy.sleep_on_exit:
            try:
                reachy.goto_sleep()
                reachy.disable_motors()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTSTP, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGHUP, _cleanup)

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading...[/bold]")

    stt = STT(
        model=config.stt.model, device=config.stt.device,
        compute_type=config.stt.compute_type, language=config.stt.language,
        beam_size=config.stt.beam_size,
    )
    stt.load()
    console.print(f"  ✓ STT (faster-whisper, {config.stt.model})")
    console.print("    CUDA warmup...", end=" ")
    console.print(f"done ({warmup_stt(stt):.1f}s)")

    silero_model = None
    if config.vad.use_silero:
        silero_model = load_silero(console)
    else:
        console.print("  [dim]Silero VAD disabled, using energy-only VAD[/dim]")

    vision_system_prompt = config.vision.system_prompt
    vision_few_shot = config.vision.few_shot or []
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=vision_system_prompt,
    )
    llm.load()
    console.print(f"  ✓ VLM ({llm.model})")

    tts = create_tts(
        backend=config.tts.backend, voice=config.tts.voice,
        speed=config.tts.speed, piper_voice=config.tts.piper_voice,
        lang=config.tts.lang,
    )
    tts = tts if tts.load() else None
    if tts:
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
    else:
        console.print("  ⚠ TTS unavailable")

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32 if silero_model else config.vad.chunk_ms
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

    n_frames = config.vision.frames
    n_fewshot = len(vision_few_shot) // 2
    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query{f', {n_fewshot} few-shot pairs' if n_fewshot else ''})[/green bold]\n"
    )

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            t_cam = time.perf_counter()
            captured_frames = cam.get_speech_frames(
                speech_start=segment.start_time,
                speech_end=segment.end_time,
                max_frames=n_frames,
            )
            dt_cam = time.perf_counter() - t_cam

            t_stt = time.perf_counter()
            result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
            text = result.get("text", "").strip()
            dt_stt = time.perf_counter() - t_stt

            if not text:
                err = result.get("error", "")
                console.print(
                    f"[dim]  (not recognized — {segment.duration:.1f}s, "
                    f"rms={segment.rms:.4f}{', err='+err if err else ''})[/dim]"
                )
                mic.resume()
                continue

            word_count = len(text.split())
            if word_count <= 2 and "?" not in text:
                console.print(f"[dim]  (skipped filler: \"{text}\")[/dim]")
                mic.resume()
                continue

            n_imgs = len(captured_frames)
            console.print(
                f'  [green]You:[/green] "{text}" '
                f'[dim]({n_imgs} frame{"s" if n_imgs != 1 else ""} captured)[/dim]'
            )

            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, text, vision_system_prompt, mic.pa_sink,
                images_b64=captured_frames if captured_frames else None,
                few_shot=vision_few_shot if vision_few_shot else None,
                first_chunk_words=config.tts.first_chunk_words,
                max_chunk_words=config.tts.max_chunk_words,
            )
            console.print()

            timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} img from buf)"
            if ttft is not None:
                toks = len(full_resp.split())
                timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | VLM no response"
            timing += "[/dim]"
            console.print(timing)

            mic.resume()

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[yellow]Goodbye![/yellow]")
        mic.stop()
        cam.close()
        stt.unload()
        llm.unload()
        if tts:
            tts.unload()
        if reachy and config.reachy.sleep_on_exit:
            try:
                reachy.goto_sleep()
                reachy.disable_motors()
            except Exception:
                pass


if __name__ == "__main__":
    main()
