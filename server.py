"""
Phosphene — Stage 2: live scrolling diffusion band.

You draw at the right-hand inlet; a real diffusion model realises your lines
into objects and the whole field flows left, reinventing itself as the seed
drifts. Fixed loose prompt + wandering seed = "pure random riff".

Architecture:
  - the model stays hot in GPU memory
  - a generation loop runs as fast as the card allows (img2img feedback):
      shift the canvas left -> fill the new right edge with noise ->
      stamp your fresh strokes into the scribble -> one ControlNet+img2img
      pass -> that becomes the next canvas -> stream it to the browser
  - the browser scrolls smoothly at 60fps between arriving frames

Run the live server (see README for setup):
    python server.py
    then open http://localhost:8900

Headless smoke test (no browser, writes frames to ./out/live_*.jpg):
    python server.py --smoke
"""

import io
import sys
import math
import asyncio
import numpy as np
import torch
from PIL import Image, ImageDraw

DEFAULTS = dict(
    mode="scroll",   # scroll | static | radial
    step=34,         # px scrolled left per frame (scroll) / zoom amount (radial)
    strength=0.62,   # how much each frame reinvents (img2img)
    control=1.10,    # how strongly it obeys your drawn line
    brush=5,
    drift=1,         # seed increments per frame (0 = frozen seed)
    keywords="",     # live prompt context typed by the user
    steps=4,         # denoise steps per frame — the FPS lever (fewer = faster)
)

# ---- band geometry: sized to the viewer's viewport aspect ----
# We keep a constant, model-friendly pixel budget (SD1.5 degrades far from ~512),
# and match the viewport's ASPECT so the client fills the screen with no letterbox.
W = H = INLET = 0
_Y = _X = _R = None

def configure(aspect, area=250000, lo=256, hi=896):
    """Pick W,H (multiples of 8) with W*H≈area and W/H==aspect; rebuild radial coords."""
    global W, H, INLET, _Y, _X, _R
    a = max(0.42, min(2.6, float(aspect)))
    H = max(lo, min(hi, int(round(((area / a) ** 0.5) / 8)) * 8))
    W = max(lo, min(hi, int(round(((area * a) ** 0.5) / 8)) * 8))
    INLET = max(48, (int(W * 0.18) // 8) * 8)     # right-edge zone (scroll) / centre disc (radial)
    _Y, _X = np.mgrid[0:H, 0:W]
    _R = np.sqrt((_X - W / 2.0) ** 2 + (_Y - H / 2.0) ** 2)

configure(640 / 384)   # default until the client reports its viewport

PROMPT = "ornate psychedelic object, intricate surreal detail, dreamlike, vivid"
NEG = "blurry, low quality, flat, dull, watermark, text, frame, border"

BASE       = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET = "lllyasviel/control_v11p_sd15_scribble"
LCM_LORA   = "latent-consistency/lcm-lora-sdv1-5"

pipe = None


def load_pipe():
    global pipe
    if pipe is not None:
        return pipe
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        LCMScheduler,
    )
    print("[phosphene] loading ControlNet + SD1.5 img2img …", flush=True)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA)
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    # a warmup pass so the first live frame isn't a stall
    z = Image.new("RGB", (W, H), "black")
    pipe(prompt="warmup", image=z, control_image=z, num_inference_steps=4,
         guidance_scale=1.0, strength=0.5, height=H, width=W,
         generator=torch.Generator("cuda").manual_seed(0))
    torch.cuda.synchronize()
    print("[phosphene] model hot.", flush=True)
    return pipe


class Band:
    """Rolling state for one live session."""
    def __init__(self):
        self.canvas = Image.new("RGB", (W, H), "black")
        self.scribble = Image.new("L", (W, H), 0)
        self.seed = 1234
        self.params = dict(DEFAULTS)
        self.paused = False
        self.alive = True

    def resize(self):
        """Rescale live content to the current W,H after a viewport change."""
        self.canvas = self.canvas.resize((W, H), Image.BILINEAR)
        self.scribble = self.scribble.resize((W, H), Image.BILINEAR)

    def advance(self, strokes):
        """Apply the per-frame geometric move for the active canvas mode,
        seed fresh material, then stamp the user's new strokes."""
        mode = self.params.get("mode", "scroll")
        step = int(self.params["step"])

        if mode == "radial":
            # zoom outward from centre; new material is born in the centre disc
            z = 1.0 + step * 0.0016
            self.canvas = self._zoom(self.canvas, z, noise=True)
            self.scribble = self._zoom(self.scribble, z, noise=False, fade=0.90)

        elif mode == "static":
            # no geometric move — the picture stays put and evolves in place
            self.scribble = self.scribble.point(lambda p: int(p * 0.94))

        else:  # scroll
            cnv = Image.new("RGB", (W, H), "black")
            cnv.paste(self.canvas.crop((step, 0, W, H)), (0, 0))
            noise = (np.random.default_rng().integers(0, 256, (H, step, 3))).astype("uint8")
            cnv.paste(Image.fromarray(noise), (W - step, 0))
            self.canvas = cnv
            scr = Image.new("L", (W, H), 0)
            scr.paste(self.scribble.crop((step, 0, W, H)).point(lambda p: int(p * 0.90)), (0, 0))
            self.scribble = scr

        self._stamp(strokes)

    def _zoom(self, img, z, noise, fade=1.0):
        """Magnify from centre by factor z (content flows outward), then
        optionally seed noise into the centre where content is starved."""
        w, h = img.size
        big = img.resize((max(w + 2, int(w * z)), max(h + 2, int(h * z))), Image.BILINEAR)
        left = (big.width - w) // 2
        top = (big.height - h) // 2
        out = big.crop((left, top, left + w, top + h))
        if fade != 1.0:
            out = out.point(lambda p: int(p * fade))
        if noise:
            arr = np.asarray(out).astype(np.float32)
            rnd = np.random.default_rng().integers(0, 256, (h, w, 3)).astype(np.float32)
            mask = np.clip((INLET * 0.6 - _R) / (INLET * 0.55), 0.0, 1.0)[..., None]
            out = Image.fromarray((arr * (1 - mask) + rnd * mask).clip(0, 255).astype("uint8"))
        return out

    def _stamp(self, strokes, width=None):
        if not strokes:
            return
        d = ImageDraw.Draw(self.scribble)
        bw = width if width else max(2, int(self.params["brush"]))
        for seg in strokes:
            if len(seg) == 1:
                x, y = seg[0]
                d.ellipse((x - bw/2, y - bw/2, x + bw/2, y + bw/2), fill=255)
            elif len(seg) >= 2:
                d.line([tuple(p) for p in seg], fill=255, width=bw, joint="curve")

    def apply_stamp(self, s):
        """Plant a crystal seed: its shape lines go into the scribble (structure),
        and a soft noise disc is nucleated into the canvas so the diffusion has
        fresh material to bloom outward from at that spot."""
        self._stamp(s.get("segs", []), width=3)
        cx, cy, r = float(s["cx"]), float(s["cy"]), max(6.0, float(s["r"]))
        arr = np.asarray(self.canvas).astype(np.float32)
        rnd = np.random.default_rng().integers(0, 256, (H, W, 3)).astype(np.float32)
        d = np.sqrt((_X - cx) ** 2 + (_Y - cy) ** 2)
        mask = np.clip((r - d) / (r * 0.6), 0.0, 1.0)[..., None]
        self.canvas = Image.fromarray((arr * (1 - mask) + rnd * mask).clip(0, 255).astype("uint8"))

    def generate(self):
        p = self.params
        kw = str(p.get("keywords", "")).strip()
        prompt = f"{kw}, intricate, detailed, vivid, surreal, dreamlike" if kw else PROMPT
        ctrl = self.scribble.convert("RGB")
        g = torch.Generator("cuda").manual_seed(int(self.seed))
        out = pipe(
            prompt=prompt, negative_prompt=NEG,
            image=self.canvas, control_image=ctrl,
            num_inference_steps=max(2, int(p.get("steps", 4))), guidance_scale=1.2,
            strength=float(p["strength"]),
            controlnet_conditioning_scale=float(p["control"]),
            height=H, width=W, generator=g,
        ).images[0]
        self.canvas = out
        self.seed += int(p["drift"])
        return out


def encode_jpeg(img, q=82):
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return b.getvalue()


# ----------------------------- smoke test -----------------------------
def smoke(n=24):
    load_pipe()
    band = Band()
    # a synthetic doodle that keeps entering the inlet
    def fake_strokes(i):
        cx = W - INLET/2
        pts = []
        for k in range(24):
            t = k / 23
            pts.append([cx + math.sin(t*6 + i*0.4) * 40, 60 + t * (H - 120)])
        return [pts]
    import time, os
    outdir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    for i in range(n):
        band.advance(fake_strokes(i))
        img = band.generate()
        if i % 4 == 0 or i == n - 1:
            img.save(os.path.join(outdir, f"live_{i:03d}.jpg"), quality=88)
    dt = (time.time() - t0) / n
    print(f"[phosphene] {n} ticks, {dt*1000:.0f} ms/tick ({1/dt:.1f} fps), "
          f"{int(DEFAULTS['step']/dt)} px/sec scroll", flush=True)
    print(f"[phosphene] wrote sample frames -> {outdir}", flush=True)


# ----------------------------- live server -----------------------------
def make_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    import os

    app = FastAPI()
    WEB = os.path.join(os.path.dirname(__file__), "web", "index.html")

    @app.get("/")
    def index():
        return FileResponse(WEB)

    @app.on_event("startup")
    def _startup():
        load_pipe()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        band = Band()
        inbox = {"strokes": [], "stamps": [], "resize": None}

        async def receiver():
            try:
                while band.alive:
                    msg = await sock.receive_json()
                    t = msg.get("type")
                    if t == "stroke":
                        inbox["strokes"].append(msg["pts"])
                    elif t == "stamp":
                        inbox["stamps"].append(msg)
                    elif t == "resize":
                        inbox["resize"] = msg.get("aspect")
                    elif t == "params":
                        band.params.update(msg["params"])
                    elif t == "pause":
                        band.paused = bool(msg["value"])
                    elif t == "clear":
                        band.canvas = Image.new("RGB", (W, H), "black")
                        band.scribble = Image.new("L", (W, H), 0)
                        band.seed += 5000
            except WebSocketDisconnect:
                band.alive = False
            except Exception:
                band.alive = False

        async def send_hello():
            await sock.send_json({"type": "hello", "w": W, "h": H,
                                  "inlet": INLET, "step": int(band.params["step"])})

        async def sender():
            await send_hello()   # tell the client the band geometry up front
            try:
                while band.alive:
                    if band.paused:
                        await asyncio.sleep(0.04)
                        continue
                    # apply a pending viewport resize at a safe point (globals + buffers together)
                    if inbox["resize"] is not None:
                        configure(inbox["resize"]); inbox["resize"] = None
                        band.resize()
                        await send_hello()
                    strokes = inbox["strokes"]; stamps = inbox["stamps"]
                    inbox["strokes"] = []; inbox["stamps"] = []
                    band.advance(strokes)
                    for s in stamps:
                        band.apply_stamp(s)
                    img = await asyncio.to_thread(band.generate)
                    await sock.send_json({"type": "step", "step": int(band.params["step"]),
                                          "mode": band.params.get("mode", "scroll")})
                    await sock.send_bytes(encode_jpeg(img))
            except (WebSocketDisconnect, RuntimeError):
                band.alive = False

        await asyncio.gather(receiver(), sender())

    return app


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        import uvicorn
        load_pipe()
        print("[phosphene] serving on all interfaces :8900 (LAN-accessible)", flush=True)
        uvicorn.run(make_app(), host="0.0.0.0", port=8900, log_level="warning")
