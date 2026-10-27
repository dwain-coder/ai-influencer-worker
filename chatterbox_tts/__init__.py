"""Chatterbox voice cloning for ComfyUI: script + reference clip -> AUDIO plus chunk timings for captions.

Chatterbox needs transformers 5.x while InfiniteTalk needs 4.51, so it runs in its own venv
(/opt/tts-venv, see serverless/Dockerfile) as a subprocess. Models download once to HF_HOME on the network volume.
"""
import json
import os
import subprocess
import tempfile
import uuid

import folder_paths
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
TTS_PYTHON = os.environ.get("TTS_PYTHON", "/opt/tts-venv/bin/python")
LANGUAGES = ["en", "ja", "zh", "ko", "fr", "de", "es", "it", "pt", "nl", "ru", "ar", "hi"]


class ChatterboxTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Hello there!"}),
            "language": (LANGUAGES, {"default": "en"}),
            # Reference clip uploaded with the job (ComfyUI input folder): the voice to clone.
            "reference": ("STRING", {"default": "voice_ref.wav"}),
            "filename_prefix": ("STRING", {"default": "voice"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "timings")
    FUNCTION = "speak"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def speak(self, text, language, reference, filename_prefix):
        ref = folder_paths.get_annotated_filepath(reference)
        with tempfile.TemporaryDirectory() as tmp:
            text_file = os.path.join(tmp, "script.txt")
            out = os.path.join(tmp, "voice.wav")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            result = subprocess.run([TTS_PYTHON, os.path.join(HERE, "speak.py"), "--ref", ref, "--lang", language, "--text-file", text_file, "--out", out],
                                    capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                raise RuntimeError(f"Chatterbox failed: {result.stderr[-1500:]}")
            waveform, sample_rate = torchaudio.load(out)
            with open(os.path.join(tmp, "voice.json"), encoding="utf-8") as f:
                timings = json.load(f)

        # Saved as an output file so the app can fetch the timings (captions, scene cutaways).
        name = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(folder_paths.get_output_directory(), name), "w", encoding="utf-8") as f:
            json.dump(timings, f, ensure_ascii=False)
        print(f"[ChatterboxTTS] {language} {timings['duration']}s audio, load {timings['loadSeconds']}s, speak {timings['speakSeconds']}s")
        return {"ui": {"captions": [{"filename": name, "subfolder": "", "type": "output"}]},
                "result": ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, json.dumps(timings, ensure_ascii=False))}


NODE_CLASS_MAPPINGS = {"ChatterboxTTS": ChatterboxTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"ChatterboxTTS": "Chatterbox TTS (voice clone)"}
