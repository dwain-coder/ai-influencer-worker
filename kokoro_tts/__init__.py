"""Kokoro TTS for ComfyUI: script text -> AUDIO plus word timings for captions.

Installed on the Pod by scripts/runpod/setup.sh (copied into ComfyUI/custom_nodes).
"""
import json
import os
import uuid

import numpy as np
import torch
import folder_paths

SAMPLE_RATE = 24000
VOICES = ["am_puck", "am_michael", "am_fenrir", "am_adam", "am_echo", "am_eric", "am_liam", "am_onyx",
          "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_river", "af_jessica",
          "bm_george", "bm_lewis", "bm_daniel", "bf_emma", "bf_isabella", "bf_alice",
          "jm_kumo", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"]  # j = Japanese (needs misaki[ja])
_pipelines = {}


def _pipeline(lang):
    from kokoro import KPipeline
    if lang not in _pipelines:
        _pipelines[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    return _pipelines[lang]


class KokoroTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Hey everyone!"}),
            "voice": (VOICES, {"default": "am_puck"}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            "filename_prefix": ("STRING", {"default": "voice"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "word_timings")
    FUNCTION = "speak"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def speak(self, text, voice, speed, filename_prefix):
        # Voice prefix letter is the language: a = American, b = British, j = Japanese.
        pipe = _pipeline(voice[0])
        chunks, words, offset = [], [], 0.0
        for result in pipe(text, voice=voice, speed=speed):
            audio = result.audio.cpu().numpy() if hasattr(result.audio, "cpu") else np.asarray(result.audio)
            for token in result.tokens or []:
                if token.start_ts is None or not token.text.strip():
                    continue
                words.append({"text": token.text, "start": round(offset + token.start_ts, 3),
                              "end": round(offset + (token.end_ts or token.start_ts), 3)})
            chunks.append(audio)
            offset += len(audio) / SAMPLE_RATE
        waveform = torch.from_numpy(np.concatenate(chunks)).float().unsqueeze(0).unsqueeze(0)
        timings = {"duration": round(offset, 3), "words": words}

        # Saved so the app can fetch timings as a regular output file.
        name = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(folder_paths.get_output_directory(), name), "w") as f:
            json.dump(timings, f)
        return {"ui": {"captions": [{"filename": name, "subfolder": "", "type": "output"}]},
                "result": ({"waveform": waveform, "sample_rate": SAMPLE_RATE}, json.dumps(timings))}


NODE_CLASS_MAPPINGS = {"KokoroTTS": KokoroTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"KokoroTTS": "Kokoro TTS"}
