import os
from bson import ObjectId
from pymongo import MongoClient, DESCENDING

_db = None


def init_db(app):
    global _db
    uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _db = client["sl_bench"]

    # Ensure indexes
    _db.runs.create_index([("created_at", DESCENDING)])
    _db.runs.create_index("status")

    app.config["DB"] = _db
    print(f"[db] Connected to MongoDB at {uri}", flush=True)


def get_db():
    return _db


def serialize_run(run: dict) -> dict:
    """Convert a MongoDB document to a JSON-safe dict."""
    if run is None:
        return None
    run = dict(run)
    run["_id"] = str(run["_id"])
    # Convert any remaining ObjectId values
    for k, v in run.items():
        if isinstance(v, ObjectId):
            run[k] = str(v)
    return run