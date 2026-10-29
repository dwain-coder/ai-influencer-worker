"""Qwen3-TTS for ComfyUI: script -> AUDIO plus chunk timings for captions.

Qwen3TTS clones a voice from a reference clip (Base model). Qwen3TTSPreset speaks with a built-in speaker
such as Ryan or Aiden, steered by a plain-language instruction (CustomVoice model).

Qwen3-TTS (Apache-2.0) pins transformers 4.57 while InfiniteTalk needs 4.51, so it runs in its own venv
(/opt/qwen-tts-venv, see Dockerfile) as a subprocess, like chatterbox_tts. Models download once to HF_HOME
on the network volume.
"""
import json
import os
import subprocess
import tempfile
import uuid

import folder_paths
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
TTS_PYTHON = os.environ.get("QWEN_TTS_PYTHON", "/opt/qwen-tts-venv/bin/python")
LANGUAGES = ["English", "Japanese", "Chinese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
CLONE_MODELS = ["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"]
PRESET_MODELS = ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]
SPEAKERS = ["Ryan", "Aiden", "Ono_Anna", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Sohee"]


def run_speak(text, language, model, filename_prefix, **files):
    """files: flag -> text written to a temp file (ref-text-file, instruct-file) or passed as-is (ref, speaker)."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [TTS_PYTHON, os.path.join(HERE, "speak.py"), "--lang", language, "--model", model]
        for flag, value in {"text-file": text, **files}.items():
            if flag.endswith("-file"):
                path = os.path.join(tmp, flag + ".txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(value)
                value = path
            cmd += [f"--{flag}", value]
        out = os.path.join(tmp, "voice.wav")
        result = subprocess.run(cmd + ["--out", out], capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"Qwen3-TTS failed: {result.stderr[-1500:]}")
        waveform, sample_rate = torchaudio.load(out)
        with open(os.path.join(tmp, "voice.json"), encoding="utf-8") as f:
            timings = json.load(f)

    # Saved as an output file so the app can fetch the timings (captions, scene cutaways).
    name = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.json"
    with open(os.path.join(folder_paths.get_output_directory(), name), "w", encoding="utf-8") as f:
        json.dump(timings, f, ensure_ascii=False)
    print(f"[Qwen3TTS] {language} {timings['duration']}s audio, load {timings['loadSeconds']}s, speak {timings['speakSeconds']}s")
    return {"ui": {"captions": [{"filename": name, "subfolder": "", "type": "output"}]},
            "result": ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, json.dumps(timings, ensure_ascii=False))}


class Qwen3TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Hello there!"}),
            "language": (LANGUAGES, {"default": "English"}),
            # Reference clip uploaded with the job (ComfyUI input folder): the voice to clone.
            "reference": ("STRING", {"default": "voice_ref.wav"}),
            # Transcript of the reference clip; empty = x-vector only (no transcript needed, slightly less faithful).
            "reference_text": ("STRING", {"multiline": True, "default": ""}),
            "model": (CLONE_MODELS, {"default": CLONE_MODELS[0]}),
            "filename_prefix": ("STRING", {"default": "voice"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "timings")
    FUNCTION = "speak"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def speak(self, text, language, reference, reference_text, model, filename_prefix):
        return run_speak(text, language, model, filename_prefix,
                         **{"ref": folder_paths.get_annotated_filepath(reference), "ref-text-file": reference_text})


class Qwen3TTSPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Hello there!"}),
            "language": (LANGUAGES, {"default": "English"}),
            "speaker": (SPEAKERS, {"default": "Ryan"}),
            # Optional delivery direction, e.g. "Calm and cinematic, documentary narrator".
            "instruct": ("STRING", {"multiline": True, "default": ""}),
            "model": (PRESET_MODELS, {"default": PRESET_MODELS[0]}),
            "filename_prefix": ("STRING", {"default": "voice"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "timings")
    FUNCTION = "speak"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def speak(self, text, language, speaker, instruct, model, filename_prefix):
        return run_speak(text, language, model, filename_prefix, **{"speaker": speaker, "instruct-file": instruct})


NODE_CLASS_MAPPINGS = {"Qwen3TTS": Qwen3TTS, "Qwen3TTSPreset": Qwen3TTSPreset}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen3TTS": "Qwen3-TTS (voice clone)", "Qwen3TTSPreset": "Qwen3-TTS (built-in speaker)"}
