#!/usr/bin/env python3
"""
AuK (Audio Universal Kernel) In-Place Speech Error-Healing & Acoustic Directing Daemon
Runs on NVIDIA Blackwell GPUs (default port :3950)
Exposes low-latency HTTP REST endpoints for surgical narration repair, pitch/timbre correction,
and paralinguistic directing in OmniVoice Studio.
"""

from __future__ import annotations
import os
import sys
import io
import time
import tempfile
import logging
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import uvicorn

from auk.infer.infer_auk import AukInfer, get_gen_duration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auk_server")

app = FastAPI(title="AuK Speech Repair & Directing Service", version="1.1")

CKPT_PATH = os.environ.get("AUK_CKPT_PATH", os.path.join(ROOT, "ckpts", "AuK", "auk_base.safetensors"))
CONFIG_PATH = os.environ.get("AUK_CONFIG_PATH", os.path.join(ROOT, "ckpts", "AuK", "config.yaml"))
QWEN_PATH = os.environ.get("AUK_QWEN_PATH", None)
CPU_OFFLOAD = os.environ.get("AUK_CPU_OFFLOAD", "0").strip().lower() in ("1", "true", "yes")

engine: Optional[AukInfer] = None


def get_engine() -> AukInfer:
    global engine
    if engine is None:
        logger.info("Initializing resident AukInfer on GPU (bf16)...")
        t0 = time.time()
        engine = AukInfer(
            config_path=CONFIG_PATH,
            ckpt_path=CKPT_PATH,
            device=os.environ.get("AUK_DEVICE", "cuda:0"),
            dtype=os.environ.get("AUK_DTYPE", "bf16"),
            qwen_path=QWEN_PATH,
            cpu_offload=CPU_OFFLOAD,
        )
        logger.info(f"AuK resident engine initialized in {time.time() - t0:.2f}s!")
    return engine


@app.on_event("startup")
def startup_event():
    logger.info("Starting AuK daemon, loading models into GPU memory...")
    try:
        get_engine()
        logger.info("✓ AuK model loaded and resident in VRAM. Ready on port 3950.")
    except Exception as e:
        logger.error(f"Failed to pre-load AuK model: {e}", exc_info=True)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine": "AuK-Base",
        "device": os.environ.get("AUK_DEVICE", "cuda:0"),
        "loaded": engine is not None,
        "target_sample_rate": 24000,
    }


@app.post("/repair")
async def repair_audio(
    file: UploadFile = File(...),
    target_text: str = Form(""),
    asr_text: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    nfe: int = Form(32),
    cfg: float = Form(2.0),
    speed: float = Form(1.0),
    preserve_duration: bool = Form(False),
):
    """
    In-place speech error healing and acoustic/paralinguistic directing.
    - Content editing (`preserve_duration=False`): scales output duration via `get_gen_duration()`
      for word replacement, insertion, or stutter deletion.
    - Same-length acoustic/paralinguistic editing (`preserve_duration=True`): omits `gen_seconds`
      per `docs/COOKBOOK.md` (Sections 3.1, 4.1, 4.5, 5.1) so pitch shifts, emotion changes,
      whisper conversion, and speech restoration preserve exact source frame duration.
    """
    t_start = time.time()
    eng = get_engine()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    with io.BytesIO(contents) as bio:
        data, sr = sf.read(bio, dtype="float32")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
        sf.write(tmp_in_path, data, sr)

    try:
        if not instruction or not instruction.strip():
            instruction = f"Fix the pronunciation so the speaker clearly and naturally says: '{target_text.strip()}'."

        content = [
            {"type": "text", "text": instruction},
            {"type": "audio", "audio": tmp_in_path},
        ]
        messages = [{"role": "user", "content": content}]

        if preserve_duration:
            gen_seconds = None
        else:
            gen_seconds = get_gen_duration(
                audio=tmp_in_path,
                ref_text=asr_text if asr_text else None,
                gen_text=target_text if target_text else None,
                speed=speed,
            )
            if gen_seconds is None or gen_seconds <= 0:
                gen_seconds = max(0.5, len(data) / sr)

        logger.info(f"Running AuK repair: instruction='{instruction[:70]}', gen_seconds={gen_seconds}")

        audio_out, out_sr = eng.generate(
            messages,
            audio=tmp_in_path,
            gen_seconds=gen_seconds,
            nfe=nfe,
            cfg_strength=cfg,
        )

        out_buf = io.BytesIO()
        out_data = audio_out.detach().cpu().to(torch.float32).squeeze(0).numpy()
        sf.write(out_buf, out_data, out_sr, format="WAV", subtype="PCM_16")
        out_bytes = out_buf.getvalue()

        elapsed = time.time() - t_start
        logger.info(f"✓ AuK repair complete in {elapsed:.2f}s ({len(out_bytes)} bytes)")

        return Response(
            content=out_bytes,
            media_type="audio/wav",
            headers={
                "X-Inference-Time": f"{elapsed:.3f}",
                "X-Sample-Rate": str(out_sr),
            },
        )
    except Exception as e:
        logger.error(f"AuK repair failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3950))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
