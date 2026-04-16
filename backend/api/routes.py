from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from flask import Blueprint, current_app, jsonify, request

from core.db import get_db, serialize_run

api_bp = Blueprint("api", __name__)

VALID_ATTACKS = {"FORA", "FSHA", "Inverse Network"}
VALID_DEFENSES = {"None", "NoPeekNN", "DP-Gaussian", "DP-Laplace", "AFO"}
VALID_ARCHITECTURES = {"Vanilla SL", "U-Shaped SL", "SplitFed"}


# ── Helper ────────────────────────────────────────────────────────────────────

def _oid(run_id: str):
    try:
        return ObjectId(run_id)
    except InvalidId:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@api_bp.get("/runs")
def list_runs():
    """Return all runs, newest first. Optional ?status= filter."""
    db = get_db()
    filt = {}
    if status := request.args.get("status"):
        filt["status"] = status
    runs = list(db.runs.find(filt).sort("created_at", -1).limit(500))
    return jsonify([serialize_run(r) for r in runs])


@api_bp.post("/runs")
def submit_run():
    """
    Submit a new experiment run. Body (JSON):
      attack        str   required
      defense       str   required
      architecture  str   default "Vanilla SL"
      cut_layer     int   default 1
      epochs        int   default 10
      note          str   optional
    Returns 202 with {run_id, status: "pending"}.
    """
    body = request.get_json(force=True, silent=True) or {}

    attack = body.get("attack", "")
    defense = body.get("defense", "")
    architecture = body.get("architecture", "Vanilla SL")
    note = body.get("note", "")

    try:
        cut_layer = int(body.get("cut_layer", 1))
        epochs = int(body.get("epochs", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "cut_layer and epochs must be integers"}), 400

    errors = []
    if attack not in VALID_ATTACKS:
        errors.append(f"attack must be one of {sorted(VALID_ATTACKS)}")
    if defense not in VALID_DEFENSES:
        errors.append(f"defense must be one of {sorted(VALID_DEFENSES)}")
    if architecture not in VALID_ARCHITECTURES:
        errors.append(f"architecture must be one of {sorted(VALID_ARCHITECTURES)}")
    if cut_layer not in (1, 2, 3):
        errors.append("cut_layer must be 1, 2, or 3")
    if not (1 <= epochs <= 200):
        errors.append("epochs must be between 1 and 200")
    if errors:
        return jsonify({"errors": errors}), 400

    db = get_db()
    doc = {
        "attack": attack,
        "defense": defense,
        "architecture": architecture,
        "cut_layer": cut_layer,
        "epochs": epochs,
        "note": note,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "finished_at": None,
        # Metrics — null until complete
        "ssim": None,
        "psnr": None,
        "dcor": None,
        "accuracy": None,
        "error": None,
        "raw_output": None,
    }
    result = db.runs.insert_one(doc)
    run_id = str(result.inserted_id)

    runner = current_app.config["RUNNER"]
    runner.enqueue(run_id, {
        "attack": attack,
        "defense": defense,
        "architecture": architecture,
        "cut_layer": cut_layer,
        "epochs": epochs,
    })

    return jsonify({"run_id": run_id, "status": "pending"}), 202


@api_bp.get("/runs/<run_id>")
def get_run(run_id):
    """Poll a single run by its MongoDB _id."""
    oid = _oid(run_id)
    if oid is None:
        return jsonify({"error": "invalid run_id"}), 400
    run = get_db().runs.find_one({"_id": oid})
    if run is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(serialize_run(run))


@api_bp.delete("/runs/<run_id>")
def delete_run(run_id):
    """Delete a run. Cannot delete a currently-running job."""
    oid = _oid(run_id)
    if oid is None:
        return jsonify({"error": "invalid run_id"}), 400
    db = get_db()
    run = db.runs.find_one({"_id": oid})
    if run is None:
        return jsonify({"error": "not found"}), 404
    if run.get("status") == "running":
        return jsonify({"error": "cannot delete a run that is currently executing"}), 409
    db.runs.delete_one({"_id": oid})
    return jsonify({"deleted": run_id})


@api_bp.get("/status")
def runner_status():
    """
    Returns the runner's current state:
      current  dict|null  — the in-progress run, or null
      queued   int        — number of pending jobs behind it
    """
    runner = current_app.config["RUNNER"]
    return jsonify(runner.status())