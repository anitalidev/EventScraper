"""Central configuration — all tunables in one place."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
# Change to any newer model (e.g. "gpt-5.5") via env var or here.
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ── Pipeline ────────────────────────────────────────────────────────────────
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "8"))
BATCH_MAX_SIZE: int = int(os.environ.get("BATCH_MAX_SIZE", "50"))
# Conservative, model-independent proxy for input tokens (usually ~4 chars/token).
BATCH_MAX_INPUT_CHARS: int = int(
    os.environ.get("BATCH_MAX_INPUT_CHARS", "60000")
)
OCR_ENABLED: bool = os.environ.get("OCR_ENABLED", "true").lower() == "true"
APP_TIMEZONE: str = os.environ.get("APP_TIMEZONE", "America/Vancouver")
# B.C. Pacific time is permanently UTC-7 as of March 8, 2026.
APP_UTC_OFFSET_HOURS: int = int(os.environ.get("APP_UTC_OFFSET_HOURS", "-7"))

# ── Confidence thresholds ───────────────────────────────────────────────────
CONFIDENCE_PUBLISH: float = 0.85  # auto-publish above this
CONFIDENCE_REVIEW: float = 0.55  # flag for review above this, reject below

# ── Storage paths ───────────────────────────────────────────────────────────
DATA_DIR: str = os.environ.get("DATA_DIR", "./data")
DB_PATH: str = os.path.join(DATA_DIR, "events.db")
IMG_DIR: str = os.path.join(DATA_DIR, "images")
RAW_DIR: str = os.path.join(DATA_DIR, "raw")

# ── Instagram scraper ────────────────────────────────────────────────────────
IG_APP_ID: str = "936619743392459"
IG_POST_COUNT: int = int(os.environ.get("IG_POST_COUNT", "12"))
IG_PAGE_SIZE: int = 12
UA: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
)

# ── UBC Discovery integration ───────────────────────────────────────────────
UBC_DISCOVERY_API_URL: str = os.environ.get("UBC_DISCOVERY_API_URL", "")
UBC_DISCOVERY_API_KEY: str = os.environ.get("UBC_DISCOVERY_API_KEY", "")

EVENT_TYPES = [
    "Workshop",
    "Networking",
    "Career",
    "Social",
    "Academic",
    "Sports",
    "Wellness",
    "Volunteer",
    "Arts",
    "Culture",
    "Food",
    "Other",
]
