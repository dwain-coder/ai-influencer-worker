# RunPod Serverless worker: ComfyUI + Wan/InfiniteTalk + Kokoro TTS + IPAdapter + VHS.
# Models are NOT in the image: they live on the network volume (/runpod-volume/ComfyUI/models).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common git ffmpeg espeak-ng libgl1 libglib2.0-0 curl ca-certificates build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv

RUN pip install --upgrade pip \
    && pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Pinned to the commits verified on the H100 pod (26 Oct 2026).
ARG COMFY_REF=cdf272f35143
RUN git clone https://github.com/comfyanonymous/ComfyUI /comfyui && cd /comfyui && git checkout ${COMFY_REF} \
    && pip install -r requirements.txt

RUN set -e; cd /comfyui/custom_nodes; \
    node() { git clone https://github.com/$1 && (cd $(basename $1) && git checkout $2); \
             [ -f $(basename $1)/requirements.txt ] && pip install -r $(basename $1)/requirements.txt || true; }; \
    node cubiq/ComfyUI_IPAdapter_plus a0f451a5113c; \
    node kijai/ComfyUI-WanVideoWrapper 088128b22424; \
    node kijai/ComfyUI-KJNodes 37ca7aff7ed1; \
    node Kosinkadink/ComfyUI-VideoHelperSuite 4d907bee61e9

# transformers >=4.52 returns no hidden_states from wav2vec2 (breaks InfiniteTalk); pins go last.
RUN pip install 'kokoro>=0.9.4' soundfile sageattention==1.0.6 runpod requests \
    && pip install 'transformers==4.51.3' 'diffusers==0.35.2' 'huggingface_hub<1'

COPY kokoro_tts /comfyui/custom_nodes/kokoro_tts
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
