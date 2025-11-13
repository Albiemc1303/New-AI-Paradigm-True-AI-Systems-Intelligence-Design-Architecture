from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import hashlib
import json
import os

API_TOKEN = os.getenv("PROVENANCE_API_TOKEN")

app = FastAPI(title="Provenance Engine v1.0")


class ProvenanceRequest(BaseModel):
    repo: str
    branch: str
    commit: str
    pr_number: int | None = None
    author: str
    timestamp: str
    files_changed: list[str]
    diff: str
    metadata: dict | None = None


# --- internal function: log the decision ---
def log_decision(request: ProvenanceRequest, status: str, reason: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "reason": reason,
        "commit": request.commit,
        "author": request.author,
        "files_changed": request.files_changed,
    }

    # Generate a unique log filename by hashing the commit + time
    name = hashlib.sha256(
        f"{request.commit}{datetime.utcnow()}".encode()
    ).hexdigest()[:16]

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{name}.json", "w") as f:
        json.dump(log_entry, f, indent=4)

    return name


# --- MAIN ENDPOINT ---
@app.post("/provenance/check")
async def provenance_check(
    payload: ProvenanceRequest,
    authorization: str = Header(None)
):
    # 1. Verify the API token
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    # 2. Rule 1: Block direct pushes to main
    if payload.branch == "main" and payload.pr_number is None:
        log = log_decision(payload, "fail", "Direct pushes to main are forbidden.")
        return {
            "status": "fail",
            "reason": "Direct pushes to main are not allowed.",
            "log_id": log
        }

    # 3. Rule 2: Mandatory diff content
    if len(payload.diff.strip()) == 0:
        log = log_decision(payload, "fail", "No diff provided.")
        return {
            "status": "fail",
            "reason": "Missing diff — provenance cannot be validated.",
            "log_id": log
        }

    # 4. Rule 3: Example policy — no changes to protected folder
    protected_paths = ["cosmic_laws", "system_core"]
    for f in payload.files_changed:
        if any(f.startswith(p) for p in protected_paths):
            log = log_decision(payload, "fail", f"Attempt to modify protected file: {f}")
            return {
                "status": "fail",
                "reason": f"Protected path modified: {f}",
                "log_id": log
            }

    # All checks passed
    log = log_decision(payload, "pass", "All rules satisfied.")
    return {
        "status": "pass",
        "reason": "All provenance rules validated.",
        "log_id": log
    }
