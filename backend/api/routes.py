"""
backend/api/routes.py

All API routes consumed by the React frontend.

Endpoints:
GET    /api/runs                  List runs (filterable, paginated)
POST   /api/runs                  Submit a new experiment run
GET    /api/runs/export.csv       Download all complete runs as CSV
GET    /api/runs/<id>             Get a single run (use for polling)
PATCH  /api/runs/<id>             Update a run's note
DELETE /api/runs/<id>             Delete one run
DELETE /api/runs                  Bulk-delete by status (e.g. ?status=failed)

GET    /api/stats                 Aggregated metrics grouped by condition
GET    /api/status                Runner state (current job + queue depth)
GET    /api/schema                Valid values for attack/defense/architecture dropdowns
"""

import csv
import io
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from flask import Blueprint, Response, current_app, jsonify, request

from core.db import get_db, serialize_run

api_bp = Blueprint("api", __name__)

VALID_ATTACKS       = ["FORA", "FSHA", "Inverse Network", "PCAT"]
VALID_DEFENSES      = ["None", "NoPeekNN", "DP-Gaussian", "DP-Laplace", "AFO"]
VALID_ARCHITECTURES = ["Vanilla SL", "U-Shaped SL", "SplitFed"]
VALID_STATUSES      = ["pending", "running", "complete", "failed"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _oid(run_id: str):
    try:
        return ObjectId(run_id)
    except (InvalidId, TypeError):
        return None


def _round(v, n=4):
    return round(v, n) if v is not None else None


# ── Run list ──────────────────────────────────────────────────────────────────

@api_bp.get("/runs")
def list_runs():
    """
    Return runs newest-first.

    Query params (all optional):
      attack        filter by attack name
      defense       filter by defense name
      architecture  filter by architecture
      status        filter by status
      limit         max results (default 500, max 1000)
    """
    db   = get_db()
    filt = {}

    for field in ("attack", "defense", "architecture", "status"):
        if val := request.args.get(field):
            filt[field] = val

    try:
        limit = min(int(request.args.get("limit", 500)), 1000)
    except (TypeError, ValueError):
        limit = 500

    runs = list(db.runs.find(filt).sort("created_at", -1).limit(limit))
    return jsonify([serialize_run(r) for r in runs])


# ── Submit run ────────────────────────────────────────────────────────────────

@api_bp.post("/runs")
def submit_run():
    """
    Queue a new experiment run.

    Body (JSON):
      attack        str  required — one of VALID_ATTACKS
      defense       str  required — one of VALID_DEFENSES
      architecture  str  default "Vanilla SL"
      cut_layer     int  default 2  (1, 2, or 3)
      epochs        int  default 15 (1–200)
      note          str  optional free-text label

    Returns 202: { run_id, status: "pending" }
    """
    body = request.get_json(force=True, silent=True) or {}

    attack       = body.get("attack", "")
    defense      = body.get("defense", "")
    architecture = body.get("architecture", "Vanilla SL")
    note         = str(body.get("note", ""))

    try:
        cut_layer = int(body.get("cut_layer", 2))
        epochs    = int(body.get("epochs", 15))
    except (TypeError, ValueError):
        return jsonify({"errors": ["cut_layer and epochs must be integers"]}), 400

    errors = []
    if attack not in VALID_ATTACKS:
        errors.append(f"attack must be one of {VALID_ATTACKS}")
    if defense not in VALID_DEFENSES:
        errors.append(f"defense must be one of {VALID_DEFENSES}")
    if architecture not in VALID_ARCHITECTURES:
        errors.append(f"architecture must be one of {VALID_ARCHITECTURES}")
    if cut_layer not in (1, 2, 3):
        errors.append("cut_layer must be 1, 2, or 3")
    if not (1 <= epochs <= 200):
        errors.append("epochs must be between 1 and 200")
    if errors:
        return jsonify({"errors": errors}), 400

    db  = get_db()
    doc = {
        "attack":       attack,
        "defense":      defense,
        "architecture": architecture,
        "cut_layer":    cut_layer,
        "epochs":       epochs,
        "note":         note,
        "status":       "pending",
        "created_at":   datetime.now(timezone.utc),
        "started_at":   None,
        "finished_at":  None,
        "ssim":         None,
        "psnr":         None,
        "dcor":         None,
        "accuracy":     None,
        "error":        None,
        "raw_output":   None,
    }
    result = db.runs.insert_one(doc)
    run_id = str(result.inserted_id)

    current_app.config["RUNNER"].enqueue(run_id, {
        "attack":       attack,
        "defense":      defense,
        "architecture": architecture,
        "cut_layer":    cut_layer,
        "epochs":       epochs,
    })

    return jsonify({"run_id": run_id, "status": "pending"}), 202


# ── CSV export ────────────────────────────────────────────────────────────────

@api_bp.get("/runs/export.csv")
def export_csv():
    """
    Download all complete runs as a CSV file.
    Useful for dropping results into a spreadsheet or LaTeX table.
    """
    db   = get_db()
    runs = list(db.runs.find({"status": "complete"}).sort("created_at", -1))

    fields = [
        "_id", "attack", "defense", "architecture", "cut_layer", "epochs",
        "ssim", "psnr", "dcor", "accuracy", "note",
        "created_at", "started_at", "finished_at",
    ]

    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for run in runs:
        writer.writerow(serialize_run(run))

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sl_bench_runs.csv"},
    )


# ── Single run ────────────────────────────────────────────────────────────────

@api_bp.get("/runs/<run_id>")
def get_run(run_id):
    """
    Fetch one run by MongoDB _id.
    Poll this endpoint every few seconds while status is 'pending' or 'running'.
    """
    oid = _oid(run_id)
    if oid is None:
        return jsonify({"error": "invalid run_id"}), 400
    run = get_db().runs.find_one({"_id": oid})
    if run is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(serialize_run(run))


@api_bp.patch("/runs/<run_id>")
def update_run(run_id):
    """
    Update mutable fields on a completed or failed run.
    Currently only 'note' is editable after the fact.

    Body: { "note": "string" }
    """
    oid = _oid(run_id)
    if oid is None:
        return jsonify({"error": "invalid run_id"}), 400

    db  = get_db()
    run = db.runs.find_one({"_id": oid})
    if run is None:
        return jsonify({"error": "not found"}), 404
    if run.get("status") == "running":
        return jsonify({"error": "cannot edit a run that is currently executing"}), 409

    body    = request.get_json(force=True, silent=True) or {}
    updates = {}
    if "note" in body:
        updates["note"] = str(body["note"])[:500]   # hard cap

    if not updates:
        return jsonify({"error": "no updatable fields in request body"}), 400

    db.runs.update_one({"_id": oid}, {"$set": updates})
    return jsonify(serialize_run(db.runs.find_one({"_id": oid})))


@api_bp.delete("/runs/<run_id>")
def delete_run(run_id):
    """Delete one run. Refuses if the run is currently executing."""
    oid = _oid(run_id)
    if oid is None:
        return jsonify({"error": "invalid run_id"}), 400

    db  = get_db()
    run = db.runs.find_one({"_id": oid})
    if run is None:
        return jsonify({"error": "not found"}), 404
    if run.get("status") == "running":
        return jsonify({"error": "cannot delete a run that is currently executing"}), 409

    db.runs.delete_one({"_id": oid})
    return jsonify({"deleted": run_id})


# ── Bulk delete ───────────────────────────────────────────────────────────────

@api_bp.delete("/runs")
def bulk_delete_runs():
    """
    Delete multiple runs by status.
    Required query param: ?status=failed  (or 'pending' / 'complete')
    Running jobs are always excluded for safety.

    Returns: { "deleted": <count> }
    """
    status = request.args.get("status")
    if status not in ("failed", "pending", "complete"):
        return jsonify({
            "error": "status query param required; must be 'failed', 'pending', or 'complete'"
        }), 400

    db     = get_db()
    result = db.runs.delete_many({"status": status})
    return jsonify({"deleted": result.deleted_count})


# ── Stats ─────────────────────────────────────────────────────────────────────

@api_bp.get("/stats")
def stats():
    """
    Pre-aggregated metrics for the dashboard charts.
    Groups complete runs by (attack, defense, architecture) and computes averages.

    Response shape:
    {
      "counts": { "total", "complete", "running", "pending", "failed" },
      "by_condition": [
        {
          "attack", "defense", "architecture", "label",
          "n",
          "avg_ssim", "avg_psnr", "avg_dcor", "avg_accuracy",
          "min_ssim", "max_ssim"
        },
        ...   sorted by avg_ssim ascending (best defense first)
      ]
    }
    """
    db = get_db()

    counts = {
        "total":    db.runs.count_documents({}),
        "complete": db.runs.count_documents({"status": "complete"}),
        "running":  db.runs.count_documents({"status": "running"}),
        "pending":  db.runs.count_documents({"status": "pending"}),
        "failed":   db.runs.count_documents({"status": "failed"}),
    }

    pipeline = [
        {"$match": {"status": "complete", "ssim": {"$ne": None}}},
        {"$group": {
            "_id": {
                "attack":       "$attack",
                "defense":      "$defense",
                "architecture": "$architecture",
            },
            "n":            {"$sum": 1},
            "avg_ssim":     {"$avg": "$ssim"},
            "avg_psnr":     {"$avg": "$psnr"},
            "avg_dcor":     {"$avg": "$dcor"},
            "avg_accuracy": {"$avg": "$accuracy"},
            "min_ssim":     {"$min": "$ssim"},
            "max_ssim":     {"$max": "$ssim"},
        }},
        {"$sort": {"avg_ssim": 1}},   # lowest SSIM first = strongest defense
    ]

    by_condition = []
    for doc in db.runs.aggregate(pipeline):
        g = doc["_id"]
        by_condition.append({
            "attack":       g["attack"],
            "defense":      g["defense"],
            "architecture": g["architecture"],
            "label":        f"{g['attack']} / {g['defense']}",
            "n":            doc["n"],
            "avg_ssim":     _round(doc["avg_ssim"]),
            "avg_psnr":     _round(doc["avg_psnr"], 2),
            "avg_dcor":     _round(doc["avg_dcor"]),
            "avg_accuracy": _round(doc["avg_accuracy"], 2),
            "min_ssim":     _round(doc["min_ssim"]),
            "max_ssim":     _round(doc["max_ssim"]),
        })

    return jsonify({"counts": counts, "by_condition": by_condition})


# ── Runner status ─────────────────────────────────────────────────────────────

@api_bp.get("/status")
def runner_status():
    """
    Live runner state.
    { current: { run_id, params } | null, queued: int }
    Poll this at ~3 s intervals while a job is in progress.
    """
    return jsonify(current_app.config["RUNNER"].status())


# ── Schema ────────────────────────────────────────────────────────────────────

@api_bp.get("/schema")
def schema():
    """
    Valid option lists for frontend dropdowns.
    Keeps the frontend in sync with backend validation without hardcoding values.
    """
    return jsonify({
        "attacks":       VALID_ATTACKS,
        "defenses":      VALID_DEFENSES,
        "architectures": VALID_ARCHITECTURES,
        "cut_layers":    [1, 2, 3],
        "epoch_range":   {"min": 1, "max": 200, "default": 15},
    })
