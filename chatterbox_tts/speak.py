"""Speak a script in a cloned voice (Chatterbox) and write chunk timings for captions.

Run by the ChatterboxTTS ComfyUI node in its own venv (/opt/tts-venv) on the RunPod worker:
  /opt/tts-venv/bin/python speak.py --ref ref.wav --lang en --text-file in.txt --out out.wav

Writes out.wav (24 kHz mono) and out.json: {"duration": s, "chunks": [{"text", "start", "end"}]}.
English uses the Turbo model; any other language uses the multilingual model (23 languages, incl. Japanese).
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import torchaudio as ta

GAP_SECONDS = 0.15


def split_chunks(script: str, lang: str):
    # Sentence-ish chunks (more stable than one long prompt). Japanese ends sentences with 。！？
    pattern = r"[^。！？!?]+[。！？!?]*" if lang == "ja" else r"[^.!?]+[.!?]*"
    limit = 90 if lang == "ja" else 280
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
    parser.add_argument("--lang", default="en")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if args.lang == "en":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        speak = lambda text: model.generate(text, audio_prompt_path=args.ref)
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        speak = lambda text: model.generate(text, language_id=args.lang, audio_prompt_path=args.ref)
    load_seconds = time.time() - t0

    script = Path(args.text_file).read_text(encoding="utf-8")
    pieces, chunks, offset = [], [], 0.0
    gap = torch.zeros(1, int(model.sr * GAP_SECONDS))
    t1 = time.time()
    for text in split_chunks(script, args.lang):
        wav = speak(text).cpu()
        length = wav.shape[1] / model.sr
        chunks.append({"text": text, "start": round(offset, 3), "end": round(offset + length, 3)})
        pieces += [wav, gap]
        offset += length + GAP_SECONDS
    audio = torch.cat(pieces[:-1], dim=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(out), audio, model.sr)
    duration = audio.shape[1] / model.sr
    meta = {"duration": round(duration, 3), "chunks": chunks, "device": device, "loadSeconds": round(load_seconds, 1), "speakSeconds": round(time.time() - t1, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, **{k: meta[k] for k in ("duration", "loadSeconds", "speakSeconds")}}))


if __name__ == "__main__":
    main()
