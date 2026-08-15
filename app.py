"""
Flask API server — thin layer that delegates everything to the pipeline.
"""
import base64
import datetime as dt
import hashlib
import os
import secrets
import uuid

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for

import config
from integrations.ubc_discovery import (
    UBCDiscoveryConflict,
    UBCDiscoveryError,
    list_events as ubc_list_events,
    publish_event,
)
from models.event import ExtractedEvent
from pipeline.runner import run
from storage.store import (
    bulk_delete,
    bulk_set_status,
    create_event,
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
_secret_key_path = os.path.join(os.path.dirname(__file__), "data", ".flask_secret")
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
if os.path.exists(_secret_key_path):
    with open(_secret_key_path, "rb") as _f:
        _secret = _f.read()
else:
    _secret = os.urandom(32)
    with open(_secret_key_path, "wb") as _f:
        _f.write(_secret)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or _secret

# ── Gmail OAuth helpers ─────────────────────────────────────────────────────

_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
_GMAIL_TOKEN_PATH = os.path.join(os.path.dirname(__file__), "data", "gmail_token.json")
_GMAIL_VERIFIER_PATH = os.path.join(os.path.dirname(__file__), "data", ".gmail_verifier")
_GMAIL_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")


def _gmail_creds():
    """Return valid Gmail credentials from the stored token, or None."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        return None

    if not os.path.exists(_GMAIL_TOKEN_PATH):
        return None

    creds = Credentials.from_authorized_user_file(_GMAIL_TOKEN_PATH, _GMAIL_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(_GMAIL_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
        except Exception:
            return None
    return creds if (creds and creds.valid) else None


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


@app.route("/hub")
def hub():
    return render_template("hub.html")


@app.route("/api/hub")
def api_hub():
    """
    Returns a merged, chronologically sorted list of:
    - All EventScraper events with status review/approved/published
    - All events fetched from UBC Discovery (if configured)

    Each item has a `_source` field: "scraper" or "ubc_discovery".
    Scraper events already in UBC Discovery are flagged with `_also_in_ubc`.
    """
    scraper_events = fetch_events(status=None)
    scraper_events = [e for e in scraper_events if e["status"] in ("review", "approved", "published")]

    ubc_events_raw = ubc_list_events()

    # Build a set of UBC Discovery IDs already linked from scraper events
    linked_ubc_ids = {e["ubc_discovery_event_id"] for e in scraper_events if e.get("ubc_discovery_event_id")}

    # Normalise UBC Discovery events into a common shape
    ubc_events = []
    for ev in ubc_events_raw:
        ubc_id = str(ev.get("id", ""))
        if ubc_id in linked_ubc_ids:
            continue  # already represented by a scraper event
        ubc_events.append({
            "_source": "ubc_discovery",
            "id": ubc_id,
            "title": ev.get("title", ""),
            "date": (ev.get("event_date") or "")[:10] or None,
            "time": (ev.get("event_date") or "")[11:16] or None,
            "location": ev.get("location_name"),
            "organizer": ev.get("club_name"),
            "description": ev.get("description"),
            "source_url": ev.get("source_url"),
            "image_url": (
                ev.get("event_picture_url") or ev.get("image_url") or
                ev.get("banner_url") or ev.get("cover_image") or
                ev.get("thumbnail_url") or ev.get("image") or ev.get("thumbnail")
            ),
            "status": "published",
        })

    for ev in scraper_events:
        ev["_source"] = "scraper"
        ev["_also_in_ubc"] = bool(ev.get("ubc_discovery_event_id"))

    combined = scraper_events + ubc_events
    combined.sort(key=lambda e: (e.get("date") or "0000-00-00", e.get("time") or ""), reverse=True)

    return jsonify(combined)


@app.route("/api/events", methods=["GET"])
def api_events():
    rows = search_events(
        q=request.args.get("q", ""),
        status=request.args.get("status") or None,
        vibe=request.args.get("vibe") or None,
        limit=int(request.args.get("limit", 200)),
    )
    return jsonify(rows)


@app.route("/api/events", methods=["POST"])
def api_create_event():
    body = request.get_json(force=True) or {}
    if not body.get("title", "").strip():
        return jsonify({"error": "Title is required"}), 400
    row = create_event(body)
    return jsonify(row), 201


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


def _get_event_or_404(event_id: int):
    event = get_event(event_id)
    if event is None:
        return None, jsonify({"error": "Not found"}), 404
    return event, None, None


def _check_transition_or_400(current_status: str, target: str):
    try:
        ExtractedEvent.check_transition(current_status, target)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return None, None


@app.route("/api/events/<int:event_id>/approve", methods=["POST"])
def api_approve_event(event_id: int):
    event, err_response, err_code = _get_event_or_404(event_id)
    if err_response:
        return err_response, err_code
    err_response, err_code = _check_transition_or_400(event["status"], "approved")
    if err_response:
        return err_response, err_code
    return jsonify(set_event_status(event_id, "approved"))


@app.route("/api/events/<int:event_id>/reject", methods=["POST"])
def api_reject_event(event_id: int):
    event, err_response, err_code = _get_event_or_404(event_id)
    if err_response:
        return err_response, err_code
    err_response, err_code = _check_transition_or_400(event["status"], "rejected")
    if err_response:
        return err_response, err_code
    return jsonify(set_event_status(event_id, "rejected"))


@app.route("/api/events/<int:event_id>/review", methods=["POST"])
def api_return_to_review(event_id: int):
    event, err_response, err_code = _get_event_or_404(event_id)
    if err_response:
        return err_response, err_code
    err_response, err_code = _check_transition_or_400(event["status"], "review")
    if err_response:
        return err_response, err_code
    return jsonify(set_event_status(event_id, "review"))


@app.route("/api/events/<int:event_id>/publish", methods=["POST"])
def api_publish_event(event_id: int):
    event, err_response, err_code = _get_event_or_404(event_id)
    if err_response:
        return err_response, err_code
    err_response, err_code = _check_transition_or_400(event["status"], "published")
    if err_response:
        return err_response, err_code

    if not config.UBC_DISCOVERY_API_URL:
        row = set_event_status(event_id, "published")
        return jsonify({**row, "ubc_discovery_skipped": True})

    try:
        created = publish_event(event)
        row = record_ubc_publish(event_id, ubc_event_id=created.ubc_event_id)
        return jsonify(row)
    except UBCDiscoveryConflict as e:
        row = record_ubc_publish(event_id, ubc_event_id=e.existing_id)
        return jsonify({**row, "ubc_discovery_conflict": True})
    except (UBCDiscoveryError, Exception) as e:
        record_ubc_publish(event_id, ubc_event_id=None, error=str(e))
        return jsonify({"error": f"UBC Discovery publish failed: {e}"}), 502


@app.route("/api/events/<int:event_id>", methods=["DELETE"])
def api_delete_event(event_id: int):
    deleted = delete_event(event_id)
    if not deleted:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"deleted": event_id})


_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@app.route("/api/events/<int:event_id>/image", methods=["POST"])
def api_upload_event_image(event_id: int):
    if get_event(event_id) is None:
        return jsonify({"error": "Not found"}), 404
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"error": "No image file provided"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in _ALLOWED_IMAGE_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    os.makedirs(_IMAGES_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{ext}"
    f.save(os.path.join(_IMAGES_DIR, filename))
    image_url = f"/api/images/{filename}"
    row = update_event(event_id, {"image_url": image_url})
    return jsonify(row)


@app.route("/api/images/<path:filename>")
def api_serve_image(filename: str):
    return send_from_directory(_IMAGES_DIR, filename)


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


# ── Email Scraper routes ────────────────────────────────────────────────────

@app.route("/email-scraper")
def email_scraper():
    return render_template("email_scraper.html")


@app.route("/email-scraper/auth")
def email_scraper_auth():
    if not os.path.exists(_GMAIL_CREDENTIALS_PATH):
        return "credentials.json not found. Download it from Google Cloud Console and place it in the EventScraper directory.", 400

    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return "google-auth-oauthlib is not installed. Run: pip install google-auth-oauthlib", 500

    flow = Flow.from_client_secrets_file(
        _GMAIL_CREDENTIALS_PATH,
        scopes=_GMAIL_SCOPES,
        redirect_uri=url_for("email_scraper_callback", _external=True),
    )
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    os.makedirs(os.path.dirname(_GMAIL_VERIFIER_PATH), exist_ok=True)
    with open(_GMAIL_VERIFIER_PATH, "w") as _vf:
        _vf.write(code_verifier)
    auth_url, state = flow.authorization_url(
        access_type="offline", prompt="consent",
        code_challenge=code_challenge, code_challenge_method="S256",
    )
    session["gmail_oauth_state"] = state
    return redirect(auth_url)


@app.route("/email-scraper/callback")
def email_scraper_callback():
    if not os.path.exists(_GMAIL_CREDENTIALS_PATH):
        return "credentials.json not found.", 400

    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return "google-auth-oauthlib is not installed.", 500

    flow = Flow.from_client_secrets_file(
        _GMAIL_CREDENTIALS_PATH,
        scopes=_GMAIL_SCOPES,
        state=session.get("gmail_oauth_state"),
        redirect_uri=url_for("email_scraper_callback", _external=True),
    )
    code_verifier = None
    if os.path.exists(_GMAIL_VERIFIER_PATH):
        with open(_GMAIL_VERIFIER_PATH) as _vf:
            code_verifier = _vf.read().strip()
        os.remove(_GMAIL_VERIFIER_PATH)
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    flow.fetch_token(
        authorization_response=request.url,
        code_verifier=code_verifier,
    )
    os.makedirs(os.path.dirname(_GMAIL_TOKEN_PATH), exist_ok=True)
    with open(_GMAIL_TOKEN_PATH, "w") as f:
        f.write(flow.credentials.to_json())
    return redirect(url_for("email_scraper"))


@app.route("/api/email-scraper/status")
def api_email_scraper_status():
    creds = _gmail_creds()
    if not creds:
        return jsonify({"connected": False})
    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        return jsonify({"connected": True, "email": profile.get("emailAddress")})
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)})


@app.route("/api/email-scraper/disconnect", methods=["POST"])
def api_email_scraper_disconnect():
    if os.path.exists(_GMAIL_TOKEN_PATH):
        os.remove(_GMAIL_TOKEN_PATH)
    return jsonify({"disconnected": True})


@app.route("/api/email-scraper/extract", methods=["POST"])
def api_email_scraper_extract():
    creds = _gmail_creds()
    if not creds:
        return jsonify({"error": "Not connected to Gmail"}), 401

    body = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400

    max_results = min(int(body.get("max_results", 20)), 100)
    q = body.get("q", "")
    model = body.get("model", config.OPENAI_MODEL)
    batch_size = int(body.get("batch_size", config.BATCH_SIZE))

    try:
        from googleapiclient.discovery import build
        from scrapers.gmail import fetch_raw_posts
        from pipeline.email_runner import run_email
    except ImportError as e:
        return jsonify({"error": f"Missing dependency: {e}"}), 500

    service = build("gmail", "v1", credentials=creds)
    raw_posts, fetch_errors = fetch_raw_posts(service, max_results=max_results, q=q)

    result = run_email(raw_posts, api_key=api_key, model=model, batch_size=batch_size)
    result["errors"] = fetch_errors + result.get("errors", [])
    return jsonify(result)


@app.route("/api/email-scraper/messages")
def api_email_scraper_messages():
    creds = _gmail_creds()
    if not creds:
        return jsonify({"error": "Not connected to Gmail"}), 401

    try:
        from googleapiclient.discovery import build
    except ImportError:
        return jsonify({"error": "google-api-python-client is not installed"}), 500

    max_results = min(int(request.args.get("max_results", 20)), 100)
    q = request.args.get("q", "")

    service = build("gmail", "v1", credentials=creds)
    list_params = {"userId": "me", "maxResults": max_results}
    if q:
        list_params["q"] = q

    results = service.users().messages().list(**list_params).execute()
    raw_messages = results.get("messages", [])

    messages = []
    for msg in raw_messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in data["payload"].get("headers", [])}
        messages.append({
            "id":      msg["id"],
            "subject": headers.get("Subject", "(No Subject)"),
            "from":    headers.get("From", ""),
            "date":    headers.get("Date", ""),
            "snippet": data.get("snippet", ""),
        })

    return jsonify({"messages": messages})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
