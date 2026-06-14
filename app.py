"""
Flask API server — thin layer that delegates everything to the pipeline.
"""
import datetime as dt
import os

from flask import Flask, jsonify, render_template, request

import config
from integrations.ubc_discovery import (
    UBCDiscoveryConflict,
    UBCDiscoveryError,
    publish_event,
)
from pipeline.runner import run
from storage.store import (
    bulk_delete,
    bulk_set_status,
    delete_event,
    fetch_events,
    get_event,
    record_ubc_publish,
    search_events,
    set_event_status,
    status_counts,
    update_event,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scrape", methods=["POST"])
def scrape():
    body = request.get_json(force=True) or {}

    # ── validate input ───────────────────────────────────────────────────────
    try:
        start_date = dt.date.fromisoformat(body["start_date"])
        end_date   = dt.date.fromisoformat(body["end_date"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing dates: {e}"}), 400

    if end_date < start_date:
        return jsonify({"error": "end_date must be on or after start_date"}), 400

    channels = [c.strip() for c in (body.get("channels") or "").splitlines() if c.strip()]
    if not channels:
        return jsonify({"error": "No channels provided"}), 400

    api_key = body.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400

    model      = body.get("model") or config.OPENAI_MODEL
    batch_size = int(body.get("batch_size") or config.BATCH_SIZE)
    ocr_on     = body.get("ocr_enabled", config.OCR_ENABLED)
    if isinstance(ocr_on, str):
        ocr_on = ocr_on.lower() == "true"

    result = run(
        channels=channels,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        batch_size=batch_size,
        ocr_enabled=ocr_on,
        model=model,
    )
    return jsonify(result)


@app.route("/events")
def events():
    status = request.args.get("status")
    rows = fetch_events(status=status)
    return jsonify(rows)


# ── Review dashboard ────────────────────────────────────────────────────────

@app.route("/review")
def review():
    return render_template("review.html")


@app.route("/api/events")
def api_events():
    rows = search_events(
        q=request.args.get("q", ""),
        status=request.args.get("status") or None,
        vibe=request.args.get("vibe") or None,
        limit=int(request.args.get("limit", 200)),
    )
    return jsonify(rows)


@app.route("/api/events/counts")
def api_counts():
    return jsonify(status_counts())


@app.route("/api/events/<int:event_id>", methods=["GET"])
def api_get_event(event_id: int):
    row = get_event(event_id)
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row)


@app.route("/api/events/<int:event_id>", methods=["PATCH"])
def api_update_event(event_id: int):
    body = request.get_json(force=True) or {}
    row = update_event(event_id, body)
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row)


@app.route("/api/events/<int:event_id>/status", methods=["POST"])
def api_set_status(event_id: int):
    body = request.get_json(force=True) or {}
    new_status = body.get("status", "")

    if new_status == "published":
        return _approve_and_publish(event_id)

    try:
        row = set_event_status(event_id, new_status)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if row is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row)


def _approve_and_publish(event_id: int):
    """
    Human-approval gate: push the event to UBC Discovery, then mark it
    published locally.  If UBC Discovery is not configured, fall back to
    local-only publish so development still works.
    """
    event = get_event(event_id)
    if event is None:
        return jsonify({"error": "Not found"}), 404

    if not config.UBC_DISCOVERY_API_URL:
        # Not configured — local-only approve (dev/test mode).
        row = set_event_status(event_id, "published")
        return jsonify({**row, "ubc_discovery_skipped": True})

    try:
        created = publish_event(event)
        row = record_ubc_publish(event_id, ubc_event_id=created.ubc_event_id)
        return jsonify(row)

    except UBCDiscoveryConflict as e:
        # Already exists in UBC Discovery — treat as success, store their ID.
        row = record_ubc_publish(event_id, ubc_event_id=e.existing_id)
        return jsonify({**row, "ubc_discovery_conflict": True})

    except (UBCDiscoveryError, Exception) as e:
        # API call failed — record the error, do NOT mark published.
        record_ubc_publish(event_id, ubc_event_id=None, error=str(e))
        return jsonify({"error": f"UBC Discovery publish failed: {e}"}), 502


@app.route("/api/events/<int:event_id>", methods=["DELETE"])
def api_delete_event(event_id: int):
    deleted = delete_event(event_id)
    if not deleted:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"deleted": event_id})


@app.route("/api/events/bulk", methods=["POST"])
def api_bulk():
    body = request.get_json(force=True) or {}
    ids = body.get("ids", [])
    action = body.get("action", "status")   # "status" | "delete"
    if not ids:
        return jsonify({"error": "No ids provided"}), 400
    if action == "delete":
        count = bulk_delete([int(i) for i in ids])
        return jsonify({"deleted": count})
    new_status = body.get("status", "")
    try:
        count = bulk_set_status([int(i) for i in ids], new_status)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"updated": count})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
