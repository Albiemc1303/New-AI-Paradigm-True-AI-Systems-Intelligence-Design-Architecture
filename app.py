from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import asyncio
import base64

API_TOKEN = os.getenv("PROVENANCE_API_TOKEN")

app = FastAPI(title="Provenance Engine v1.1")


class ProvenanceRequest(BaseModel):
    repo: str
    branch: str
    commit: str
    pr_number: int | None = None
    author: str
    timestamp: str
    files_changed: list[str]
    diff: str  # Base64 encoded
    metadata: dict | None = None  # e.g., artifacts: [{path: str, hash: str}]


# --- internal function: log the decision ---
def log_decision(request: ProvenanceRequest, status: str, reason: str, diff_hash: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "reason": reason,
        "commit": request.commit,
        "author": request.author,
        "files_changed": request.files_changed,
        "diff_hash": diff_hash,
    }

    # Generate a unique log filename by hashing the commit + time
    name = hashlib.sha256(f"{request.commit}{datetime.utcnow()}".encode()).hexdigest()[:16]

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{name}.json", "w") as f:
        json.dump(log_entry, f, indent=4)

    return name


# --- internal async function for protected path check ---
async def check_protected_path(file_path: str, protected_paths: list[str]):
    normalized = os.path.normpath(file_path)
    for p in protected_paths:
        if normalized.startswith(os.path.normpath(p)):
            return file_path
    return None


@app.post("/provenance/check")
async def provenance_check(payload: ProvenanceRequest, authorization: str = Header(None)):
    # 1. Verify the API token
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    # 2. Decode the base64 diff
    try:
        diff_content = base64.b64decode(payload.diff).decode()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 diff")

    diff_hash = hashlib.sha256(diff_content.encode()).hexdigest()

    # 3. Rule 1: Block direct pushes to main
    if payload.branch == "main" and payload.pr_number is None:
        log = log_decision(payload, "fail", "Direct pushes to main are forbidden.", diff_hash)
        return {"status": "fail", "reason": "Direct pushes to main are not allowed.", "log_id": log}

    # 4. Rule 2: Mandatory diff content
    if len(diff_content.strip()) == 0:
        log = log_decision(payload, "fail", "No diff provided.", diff_hash)
        return {"status": "fail", "reason": "Missing diff — provenance cannot be validated.", "log_id": log}

    # 5. Rule 3: Protected paths (parallel)
    protected_paths = ["cosmic_laws", "system_core"]
    tasks = [check_protected_path(f, protected_paths) for f in payload.files_changed]
    results = await asyncio.gather(*tasks)
    modified_protected = [r for r in results if r is not None]
    if modified_protected:
        log = log_decision(payload, "fail", f"Attempt to modify protected files: {modified_protected}", diff_hash)
        return {
            "status": "fail",
            "reason": f"Protected paths modified: {modified_protected}",
            "log_id": log
        }

    # 6. Artifact validation (from metadata)
    artifacts = payload.metadata.get("artifacts", []) if payload.metadata else []
    for artifact in artifacts:
        path = artifact.get("path")
        expected_hash = artifact.get("hash")
        if not path or not expected_hash:
            continue
        if not Path(path).exists():
            log = log_decision(payload, "fail", f"Missing artifact: {path}", diff_hash)
            return {
                "status": "fail",
                "reason": f"Missing artifact: {path}",
                "log_id": log
            }
        actual_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            log = log_decision(payload, "fail", f"Artifact hash mismatch: {path}", diff_hash)
            return {
                "status": "fail",
                "reason": f"Artifact hash mismatch: {path}",
                "log_id": log
            }

    # All checks passed
    log = log_decision(payload, "pass", "All rules satisfied.", diff_hash)
    return {
        "status": "pass",
        "reason": "All provenance rules validated.",
        "log_id": log,
        "diff_hash": diff_hash
    }
