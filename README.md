# ai-influencer-worker

RunPod Serverless worker for the AI influencer studio: ComfyUI + Wan 2.1 InfiniteTalk + Kokoro TTS + IPAdapter + Qwen-Image-Edit + BiRefNet.
Models are not included; they are mounted from a RunPod network volume at /runpod-volume/ComfyUI/models.

See handler.py for the job API (run / fetch / manifest).
