"""Speak a script in a cloned voice (Qwen3-TTS Base) and write chunk timings for captions.

Run by the Qwen3TTS ComfyUI node in its own venv (/opt/qwen-tts-venv) on the RunPod worker:
  /opt/qwen-tts-venv/bin/python speak.py --ref ref.wav --ref-text-file ref.txt --lang English --text-file in.txt --out out.wav

Writes out.wav (mono) and out.json: {"duration": s, "chunks": [{"text", "start", "end"}]}.
The reference is encoded once (create_voice_clone_prompt) and reused; all chunks are generated in one batch.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

GAP_SECONDS = 0.15


def split_chunks(script: str, lang: str):
    # Sentence-ish chunks (more stable than one long prompt). Japanese/Chinese end sentences with 。！？
    cjk = lang in ("Japanese", "Chinese")
    pattern = r"[^。！？!?]+[。！？!?]*" if cjk else r"[^.!?]+[.!?]*"
    limit = 90 if cjk else 280
    chunks, current = [], ""
    for sentence in re.findall(pattern, script.replace("\n", " ")):
        if current and len(current) + len(sentence) > limit:
            chunks.append(current.strip())
            current = ""
        current += sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ref-text-file", required=True)
    parser.add_argument("--lang", default="English")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    prompt = model.create_voice_clone_prompt(ref_audio=args.ref, ref_text=ref_text or None, x_vector_only_mode=not ref_text)
    load_seconds = time.time() - t0

    texts = split_chunks(Path(args.text_file).read_text(encoding="utf-8"), args.lang)
    if not texts:
        raise SystemExit("Script is empty.")
    t1 = time.time()
    wavs, sr = model.generate_voice_clone(text=texts, language=[args.lang] * len(texts), voice_clone_prompt=prompt)
    gap = np.zeros(int(sr * GAP_SECONDS), dtype=np.float32)
    pieces, chunks, offset = [], [], 0.0
    for text, wav in zip(texts, wavs):
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        length = len(wav) / sr
        chunks.append({"text": text, "start": round(offset, 3), "end": round(offset + length, 3)})
        pieces += [wav, gap]
        offset += length + GAP_SECONDS
    audio = np.concatenate(pieces[:-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr)
    meta = {"duration": round(len(audio) / sr, 3), "chunks": chunks, "loadSeconds": round(load_seconds, 1), "speakSeconds": round(time.time() - t1, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, **{k: meta[k] for k in ("duration", "loadSeconds", "speakSeconds")}}))


if __name__ == "__main__":
    main()
