# Setup Guide

Full installation instructions for the Reachy Mini Jetson Assistant.

## Prerequisites

### Hardware

- **NVIDIA Jetson Orin Nano** (8GB) — other Jetson modules may work but are untested
- **[Reachy Mini Lite](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started)** — the developer version, USB connection to your computer. Provides camera, microphone, speaker, and 9-DOF motor control in one cable. [Buy Reachy Mini](https://www.hf.co/reachy-mini/)
- **NVMe SSD** recommended — for swap space and model storage

If you're new to Reachy Mini, start with the [official getting started guide](https://huggingface.co/docs/reachy_mini/index) and the [Reachy Mini Lite setup](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started). The [Python SDK documentation](https://huggingface.co/docs/reachy_mini/SDK/readme) covers movement, camera, audio, and AI integrations.

### Software

- **JetPack 6.x** (L4T r36.x, Ubuntu 22.04, CUDA 12.6)
- **Python 3.10** (ships with JetPack 6 Ubuntu 22.04)
- **Docker** with NVIDIA runtime (`nvidia-container-toolkit`)
- **PulseAudio** (for mic/speaker multiplexing)

> **Important:** This project requires **Python 3.10** specifically. The Jetson ONNX Runtime GPU wheels, CTranslate2 builds, and Reachy Mini SDK are all built against Python 3.10 on JetPack 6. Using a different Python version will cause compatibility issues.

## Hardware Setup

### Reachy Mini Lite

1. Connect Reachy Mini Lite to your Jetson via USB. The robot provides camera, microphone, speaker, and motor control over a single USB connection.

2. Add udev rules so the SDK can access the robot's serial ports without root:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", MODE="0666", SYMLINK+="reachy_mini"' \
  | sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

3. Add your user to the `dialout` group and reboot:

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

4. Verify the device is visible:

```bash
ls -la /dev/ttyACM*
# Should show /dev/ttyACM0, /dev/ttyACM1, etc.
```

### NVMe Swap (Required for 8GB Jetson)

Running STT + VLM + TTS simultaneously exceeds 8GB RAM. Setting up swap on NVMe prevents OOM kills:

```bash
sudo fallocate -l 8G /mnt/nvme/swapfile   # adjust path to your NVMe mount
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Persist across reboots:
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Installation

### Step 1: System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  portaudio19-dev \
  libasound2-dev \
  pulseaudio-utils \
  libcudnn9-dev-cuda-12
```

### Step 2: Clone and Create Virtual Environment

```bash
git clone https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant
cd reachy-mini-jetson-assistant
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Packages

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Step 4: Install ONNX Runtime GPU (Jetson-Specific)

The default `onnxruntime` from pip is CPU-only. For GPU inference (Kokoro TTS, Silero VAD) on Jetson:

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> If `CUDAExecutionProvider` isn't listed after install, uninstall the CPU version first: `pip uninstall onnxruntime && pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

### Step 5: Install Reachy Mini SDK

```bash
pip install reachy-mini
```

### Step 6: Pin NumPy (Compatibility Fix)

The Jetson `onnxruntime-gpu` wheel requires NumPy 1.x:

```bash
pip install "numpy==1.26.4"
```

### Step 7: Build CTranslate2 with CUDA (GPU-Accelerated STT)

The pip `ctranslate2` package is CPU-only. For GPU-accelerated speech-to-text on Jetson, build from source:

```bash
pip install pybind11

cd ~
git clone --depth 1 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git submodule update --init --recursive

mkdir build && cd build
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH_LIST="8.7" -DOPENMP_RUNTIME=NONE -DWITH_MKL=OFF

make -j$(nproc)
cmake --install . --prefix ~/.local

export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
cd ../python
pip install .
```

Persist the library path in your venv activation script:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/reachy-mini-jetson-assistant/venv/bin/activate
```

### Verify Installation

```bash
source venv/bin/activate
python3 -c "
import ctranslate2; print('CTranslate2 CUDA devices:', ctranslate2.get_cuda_device_count())
import onnxruntime; print('ONNX providers:', onnxruntime.get_available_providers())
from reachy_mini import ReachyMini; print('Reachy Mini SDK: OK')
import faster_whisper; print('faster-whisper: OK')
import kokoro_onnx; print('kokoro-onnx: OK')
"
```

Expected output:
```
CTranslate2 CUDA devices: 1
ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Reachy Mini SDK: OK
faster-whisper: OK
kokoro-onnx: OK
```

## Models

### LLM / VLM (served via llama.cpp Docker)

Models download automatically from HuggingFace on first launch. No manual download needed.

| Model | Use | Launch Command |
|-------|-----|----------------|
| Cosmos-Reason2-2B (Q4_K_M) | Vision VLM | `NP=1 ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M` |
| Gemma 3 1B (Q8) | Text LLM | `./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0` |
| bge-small-en-v1.5 (Q8) | RAG embeddings | `./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0` |

Models are cached in `~/.cache/huggingface` and reused across runs.

### TTS Voices

**Kokoro TTS** (default) downloads automatically on first run (~340 MB). No manual step needed.

To pre-download for offline use:

```bash
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Configure voice in `config/settings.yaml`:

```yaml
tts:
  voice: "af_sarah"    # kokoro voices: af_sarah, af_bella, am_adam, bf_emma, bm_george
```

## Troubleshooting

**`CUDAExecutionProvider` not available:**
Uninstall CPU onnxruntime and reinstall the GPU version:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**CTranslate2 not finding CUDA:**
Make sure the library path is set: `export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH`

**VLM server not responding:**
Check the Docker container is running: `docker ps`. View logs: `docker logs assistant-llm`

**Process won't exit / robot stays awake after Ctrl+C:**
The app handles Ctrl+C cleanly — the robot should go to sleep. If the process is stuck, run `pkill -9 -f run_web_vision_chat` and `pkill -f reachy-mini-daemon`.

**Port 8090 already in use:**
A previous instance is still running. Kill it: `lsof -ti :8090 | xargs kill -9`

**Camera not found:**
Check the device is available: `ls /dev/video*`. If another process holds it: `fuser -k /dev/video0`
