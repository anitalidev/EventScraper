"""
Instagram scraper.

Uses the private Instagram API endpoint that the official web app uses.
Returns RawPost objects; image downloading is handled here so CDN URLs
(which expire) are captured immediately.
"""
from __future__ import annotations
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
from models.event import RawPost


def username_from_input(raw: str) -> str:
    """Accept a full URL, @username, or bare username."""
    raw = raw.strip().rstrip("/")
    if "/" in raw:
        parts = [p for p in urllib.parse.urlparse(raw).path.split("/") if p]
        return parts[0] if parts else ""
    return raw.lstrip("@")


def _img_url(item: dict) -> Optional[str]:
    if item.get("carousel_media"):
        item = item["carousel_media"][0]
    candidates = (item.get("image_versions2") or {}).get("candidates") or []
    return candidates[0].get("url") if candidates else None


def _download_image(url: str, username: str, post_id: str) -> Optional[str]:
    if not url:
        return None
    os.makedirs(config.IMG_DIR, exist_ok=True)
    try:
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].split("?")[0].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        safe_username = re.sub(r"[^A-Za-z0-9_.-]", "_", username)
        safe_post_id = re.sub(r"[^A-Za-z0-9_.-]", "_", post_id)
        fname = f"ig_{safe_username}_{safe_post_id}{ext}"
        path = os.path.join(config.IMG_DIR, fname)
        if not os.path.exists(path):
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
        print(f"[scraper] image download failed: {e}")
        return None


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", s or "").strip())


def _post_url(username: str, item: dict) -> str:
    code = item.get("code") or item.get("shortcode")
    return f"https://www.instagram.com/p/{code}/" if code else f"https://www.instagram.com/{username}/"


def fetch_posts(
    username: str,
    start_date: date,
    end_date: date,
    download_images: bool = True,
    post_limit: int = config.IG_POST_COUNT,
) -> tuple[list[RawPost], Optional[str]]:
    """
    Fetch posts for *username* whose taken_at falls in [start_date, end_date].
    Paginates using next_max_id until all posts in the date range are collected
    or *post_limit* posts have been inspected. Returns (posts, error_message).
    """
    if post_limit < 1:
        raise ValueError("post_limit must be at least 1")

    base_url = (
        f"https://www.instagram.com/api/v1/feed/user/"
        f"{urllib.parse.quote(username)}/username/?__a=1&__d=dis"
    )
    headers = {
        "User-Agent": config.UA,
        "Accept": "application/json",
        "X-IG-App-ID": config.IG_APP_ID,
        "Referer": f"https://www.instagram.com/{username}/",
    }

    posts: list[RawPost] = []
    next_max_id: Optional[str] = None
    seen: set[str] = set()
    inspected = 0

    while inspected < post_limit:
        request_count = min(config.IG_PAGE_SIZE, post_limit - inspected)
        url = f"{base_url}&count={request_count}"
        if next_max_id:
            url += f"&max_id={urllib.parse.quote(next_max_id)}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            return posts, str(e)

        items = (data.get("items") or [])[:request_count]
        if not items:
            break

        oldest_in_page: Optional[date] = None
        for item in items:
            inspected += 1
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
            image_path = (
                _download_image(img_url, username, str(post_id))
                if download_images and img_url
                else None
            )

            posts.append(RawPost(
                source="instagram",
                username=username,
                post_url=_post_url(username, item),
                taken_at=post_dt.isoformat(),
                caption=caption,
                image_path=image_path,
            ))

        # Stop paginating once we've gone past the start of the date range
        if oldest_in_page and oldest_in_page < start_date:
            break
        if inspected >= post_limit:
            break

        next_max_id = data.get("next_max_id")
        if not next_max_id or not data.get("more_available"):
            break

        time.sleep(0.3)

    return posts, None


def scrape_channels(
    channels: list[str],
    start_date: date,
    end_date: date,
    download_images: bool = True,
    post_limit: int = config.IG_POST_COUNT,
) -> tuple[list[RawPost], list[dict]]:
    """
    Scrape all channels and return (raw_posts, errors).
    Errors are dicts with keys 'username' and 'error'.
    """
    all_posts: list[RawPost] = []
    errors: list[dict] = []

    for raw in channels:
        username = username_from_input(raw)
        if not username:
            continue
        posts, err = fetch_posts(
            username,
            start_date,
            end_date,
            download_images,
            post_limit,
        )
        if err:
            errors.append({"username": username, "error": err})
        else:
            all_posts.extend(posts)
        time.sleep(0.1)

    return all_posts, errors
