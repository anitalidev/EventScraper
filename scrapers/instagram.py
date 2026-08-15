"""
Instagram scraper.

Uses the private Instagram API endpoint that the official web app uses.
Builds a list of Source objects and delegates to the generalized pipeline.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import Optional

import config
from pipeline.event_creation import Source, create_events


def username_from_input(raw: str) -> str:
    """Accept a full URL, @username, or bare username."""
    raw = raw.strip().rstrip("/")
    if "/" in raw:
        parts = [p for p in urllib.parse.urlparse(raw).path.split("/") if p]
        return parts[0] if parts else ""
    return raw.lstrip("@")


def _api_request(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": config.UA,
        "Accept": "application/json",
        "X-IG-App-ID": config.IG_APP_ID,
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _img_url(item: dict) -> Optional[str]:
    if item.get("carousel_media"):
        item = item["carousel_media"][0]
    candidates = (item.get("image_versions2") or {}).get("candidates") or []
    return candidates[0].get("url") if candidates else None


def _download_image(url: str, username: str) -> Optional[str]:
    if not url:
        return None
    os.makedirs(config.IMG_DIR, exist_ok=True)
    try:
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].split("?")[0].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        fname = f"ig_{username}_{hashlib.sha1(url.encode()).hexdigest()[:12]}{ext}"
        path = os.path.join(config.IMG_DIR, fname)
        req = urllib.request.Request(url, headers={
            "User-Agent": config.UA,
            "Referer": "https://www.instagram.com/",
        })
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        print(f"[instagram] image download failed: {e}")
        return None


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", s or "").strip())


def _post_url(username: str, item: dict) -> str:
    code = item.get("code") or item.get("shortcode")
    return f"https://www.instagram.com/p/{code}/" if code else f"https://www.instagram.com/{username}/"


def _fetch_sources(
    username: str,
    start_date: date,
    end_date: date,
) -> tuple[list[Source], Optional[str]]:
    """
    Fetch posts for one username and return (sources, error).
    Paginates until all posts in the date range are collected.
    """
    base_url = (
        f"https://www.instagram.com/api/v1/feed/user/"
        f"{urllib.parse.quote(username)}/username/?count=12&__a=1&__d=dis"
    )
    headers = {
        "User-Agent": config.UA,
        "Accept": "application/json",
        "X-IG-App-ID": config.IG_APP_ID,
        "Referer": f"https://www.instagram.com/{username}/",
    }

    sources: list[Source] = []
    next_max_id: Optional[str] = None
    seen: set[str] = set()

    while True:
        url = base_url + (f"&max_id={urllib.parse.quote(next_max_id)}" if next_max_id else "")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            return sources, str(e)

        items = data.get("items") or []
        if not items:
            break

        oldest_in_page: Optional[date] = None
        for item in items:
            taken = item.get("taken_at") or 0
            post_dt = datetime.fromtimestamp(taken, tz=timezone.utc)
            post_date = post_dt.date()

            if oldest_in_page is None or post_date < oldest_in_page:
                oldest_in_page = post_date

            post_id = item.get("pk") or item.get("id") or ""
            if post_id in seen:
                continue
            seen.add(post_id)

            if post_date < start_date or post_date > end_date:
                continue

            caption = _clean((item.get("caption") or {}).get("text") or "")
            img_url = _img_url(item)
            image_path = _download_image(img_url, username) if img_url else None

            sources.append(Source(
                source=_post_url(username, item),
                text=caption,
                image=image_path,
            ))

        if oldest_in_page and oldest_in_page < start_date:
            break

        next_max_id = data.get("next_max_id")
        if not next_max_id or not data.get("more_available"):
            break

        time.sleep(0.3)

    return sources, None


def scrape_instagram(
    channels: list[str],
    start_date: date,
    end_date: date,
    api_key: str,
    *,
    ocr_enabled: bool = config.OCR_ENABLED,
    batch_size: int = config.BATCH_SIZE,
    model: str = config.OPENAI_MODEL,
) -> tuple[list[dict], int]:
    """
    Scrape Instagram channels and extract events.

    Returns:
        (events, posts_scraped) — list of created event rows and the count
        of posts that were successfully fetched from Instagram.
    """
    sources: list[Source] = []

    for raw in channels:
        username = username_from_input(raw)
        if not username:
            continue
        channel_sources, err = _fetch_sources(username, start_date, end_date)
        if err:
            print(f"[instagram] {username}: {err}")
        sources.extend(channel_sources)
        time.sleep(0.5)

    posts_scraped = len(sources)
    events = create_events(sources, "Instagram", api_key,
                    ocr_enabled=ocr_enabled, batch_size=batch_size, model=model)
    return events, posts_scraped
