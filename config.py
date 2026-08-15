"""Central configuration — all tunables in one place."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
# Change to any newer model (e.g. "gpt-4o-mini", "o3") via env var or here.
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ── Pipeline ────────────────────────────────────────────────────────────────
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "8"))
OCR_ENABLED: bool = os.environ.get("OCR_ENABLED", "true").lower() == "true"

# ── Confidence thresholds ───────────────────────────────────────────────────

# ── Storage paths ───────────────────────────────────────────────────────────
DATA_DIR: str = os.environ.get("DATA_DIR", "./data")
DB_PATH:  str = os.path.join(DATA_DIR, "events.db")
IMG_DIR:  str = os.path.join(DATA_DIR, "images")
RAW_DIR:  str = os.path.join(DATA_DIR, "raw")

# ── Instagram scraper ────────────────────────────────────────────────────────
IG_APP_ID:    str = "936619743392459"
IG_POST_COUNT: int = 50
UA: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
)

# ── UBC Discovery integration ───────────────────────────────────────────────
UBC_DISCOVERY_API_URL: str = os.environ.get("UBC_DISCOVERY_API_URL", "")
UBC_DISCOVERY_API_KEY: str = os.environ.get("UBC_DISCOVERY_API_KEY", "")

# ── AWS / S3 (for uploading event images on publish) ────────────────────────
AWS_REGION:     str = os.environ.get("AWS_REGION", "us-west-2")
S3_BUCKET_NAME: str = os.environ.get("S3_BUCKET_NAME", "")
S3_ENDPOINT_URL: str = os.environ.get("S3_ENDPOINT_URL", "")

EVENT_TYPES = [
    "Workshop", "Networking", "Career", "Social", "Academic",
    "Sports", "Wellness", "Volunteer", "Arts", "Culture", "Food", "Other",
]
