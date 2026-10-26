"""RunPod Serverless handler: run any ComfyUI API workflow, keep every output file.

Input:
  {"op": "run", "workflow": {...}, "files": [{"name": "x.png", "data": "<base64>"}], "timeout": 3600}
  {"op": "fetch", "job_id": "...", "filename": "...", "offset": 0}    -> one chunk of a stored output
  {"op": "manifest", "job_id": "..."}                                 -> outputs of an earlier run
Run output: {"files": [{"filename", "size", "node", "key"}], "inline": {"<filename>": "<base64>"}?}

Outputs are kept on the network volume (/runpod-volume/jobs/<job_id>/) for 24 h because RunPod caps
job results at ~10 MB; clients pull big files with "fetch" in 7 MB chunks.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import time
import uuid

import requests
import runpod

COMFY_DIR = "/comfyui"
COMFY_URL = "http://127.0.0.1:8188"
VOLUME = "/runpod-volume"
JOBS_DIR = f"{VOLUME}/jobs"
CHUNK = 7_000_000
INLINE_LIMIT = 6_000_000
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_comfy = None


def link_models():
    # Models live on the network volume; point ComfyUI's models dir at it.
    source = f"{VOLUME}/ComfyUI/models"
    target = f"{COMFY_DIR}/models"
    if os.path.isdir(source) and not os.path.islink(target):
        shutil.rmtree(target, ignore_errors=True)
        os.symlink(source, target)


def ensure_comfy():
    global _comfy
    if _comfy and _comfy.poll() is None:
        return
    link_models()
    _comfy = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"], cwd=COMFY_DIR)
    for _ in range(300):
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        if _comfy.poll() is not None:
            raise RuntimeError("ComfyUI exited during startup")
        time.sleep(1)
    raise RuntimeError("ComfyUI did not start within 300 s")


def purge_old_jobs(max_age_s=24 * 3600):
    if not os.path.isdir(JOBS_DIR):
        return
    now = time.time()
    for name in os.listdir(JOBS_DIR):
        path = os.path.join(JOBS_DIR, name)
        if now - os.path.getmtime(path) > max_age_s:
            shutil.rmtree(path, ignore_errors=True)


def job_dir(job_id):
    if not SAFE_NAME.match(job_id):
        raise ValueError("invalid job id")
    return os.path.join(JOBS_DIR, job_id)


def run(job_id, inp):
    ensure_comfy()
    purge_old_jobs()
    for item in inp.get("files", []):
        name = item["name"]
        if not SAFE_NAME.match(name):
            return {"error": f"invalid input filename: {name}"}
        with open(os.path.join(COMFY_DIR, "input", name), "wb") as f:
            f.write(base64.b64decode(item["data"].split(",")[-1]))

    response = requests.post(f"{COMFY_URL}/prompt", json={"prompt": inp["workflow"], "client_id": str(uuid.uuid4())}, timeout=30)
    body = response.json()
    if not response.ok or body.get("node_errors"):
        return {"error": "ComfyUI rejected the workflow", "details": body}
    prompt_id = body["prompt_id"]

    deadline = time.time() + int(inp.get("timeout", 3600))
    while True:
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if history and history.get("status", {}).get("completed"):
            break
        if history and history.get("status", {}).get("status_str") == "error":
            messages = [m for m in history["status"].get("messages", []) if m[0] == "execution_error"]
            return {"error": "ComfyUI execution error", "details": messages[-1][1] if messages else None}
        if time.time() > deadline:
            requests.post(f"{COMFY_URL}/interrupt", timeout=10)
            return {"error": "workflow timed out"}
        time.sleep(1)

    out_dir = job_dir(job_id)
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for node_id, node_output in (history.get("outputs") or {}).items():
        for key, items in node_output.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "filename" not in item or item.get("type") != "output":
                    continue
                src = os.path.join(COMFY_DIR, "output", item.get("subfolder", ""), item["filename"])
                if not os.path.isfile(src) or any(f["filename"] == item["filename"] for f in files):
                    continue
                shutil.move(src, os.path.join(out_dir, item["filename"]))
                files.append({"filename": item["filename"], "size": os.path.getsize(os.path.join(out_dir, item["filename"])), "node": node_id, "key": key})
    result = {"files": files}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(result, f)
    if sum(f["size"] for f in files) <= INLINE_LIMIT:
        result["inline"] = {f["filename"]: base64.b64encode(open(os.path.join(out_dir, f["filename"]), "rb").read()).decode() for f in files}
    return result


def fetch(inp):
    name = inp["filename"]
    if not SAFE_NAME.match(name):
        return {"error": "invalid filename"}
    path = os.path.join(job_dir(inp["job_id"]), name)
    if not os.path.isfile(path):
        return {"error": "output not found (expired after 24 h?)"}
    offset = int(inp.get("offset", 0))
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(CHUNK)
    return {"data": base64.b64encode(data).decode(), "offset": offset, "size": size, "eof": offset + len(data) >= size}


def manifest(inp):
    path = os.path.join(job_dir(inp["job_id"]), "manifest.json")
    if not os.path.isfile(path):
        return {"error": "manifest not found"}
    return json.load(open(path))


def handler(job):
    inp = job["input"]
    op = inp.get("op", "run")
    try:
        if op == "fetch":
            return fetch(inp)
        if op == "manifest":
            return manifest(inp)
        return run(job["id"], inp)
    except Exception as error:  # surfaced to the app as a failed job
        return {"error": f"{type(error).__name__}: {error}"}


runpod.serverless.start({"handler": handler})
