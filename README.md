# Phosphene — real-time diffusion band

Draw on a canvas and a real diffusion model turns your marks into a
**living, endlessly-scrolling psychedelic band**. Each frame the field flows
(left / outward / in place), reinventing itself as the seed drifts — running in
real time on a local NVIDIA GPU.

It's a browser front-end talking over a websocket to a Python server that keeps
the model hot and streams generated frames as fast as the card allows.

## What it does

- **Draw** at the right-hand inlet and your strokes get realised by the model,
  then flow away and evolve.
- **Three canvas modes** — `Scroll` (flows left), `Radial` (zooms outward from
  centre), `Static` (evolves in place).
- **Crystal-seed stamps** — plant glowing shapes (dendrite, lattice, starburst,
  seed-of-life, spiral, eye, face) that bloom into the field.
- **Keywords** — type words to steer what the model riffs into.
- **Rewind** — scrub back through the last ~60 seconds, frame by frame.
- **Refresh · steps** — the FPS lever: fewer denoise steps = more frames/sec.
- **Brush glow**, **Pause**, **Shot** (save a PNG), **Clear**.
- Smooth **crossfade** between renders so it feels fluid even at low frame rates.

## Requirements

- An **NVIDIA GPU** with a recent driver (developed on an RTX 4080 SUPER, 16 GB).
- **Python 3.11** (3.12+ may lag the ML wheels).
- ~6 GB of disk for PyTorch + the SD1.5 / ControlNet / LCM weights (downloaded
  from Hugging Face on first run).

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate

# PyTorch with CUDA first (pick the index matching your CUDA, e.g. cu124):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

Then open **http://localhost:8900** (the server also binds `0.0.0.0`, so other
devices on your LAN can reach it at `http://<your-ip>:8900`).

First launch downloads the models and warms the pipeline (~15 s); after that the
model stays resident and frames stream continuously.

## How it works

The server holds a `StableDiffusionControlNetImg2ImgPipeline` (SD1.5 + a scribble
ControlNet + an LCM-LoRA for few-step generation) in GPU memory. Each tick it
shifts the current frame (scroll/zoom), seeds fresh noise where new material
enters, stamps in your latest strokes as a ControlNet scribble, runs **one**
few-step img2img pass conditioned on the previous frame, and streams the result
to the browser as JPEG. The browser interpolates pan/zoom and crossfades between
frames for smoothness.

## Models

- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `lllyasviel/control_v11p_sd15_scribble`
- `latent-consistency/lcm-lora-sdv1-5`

Fetched automatically from Hugging Face on first run.
