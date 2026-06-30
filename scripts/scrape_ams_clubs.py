#!/usr/bin/env python3
"""Scrape the UBC AMS club directory and write club contacts to CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DIRECTORY_URL = "https://amsclubs.ca/all-clubs/"
DEFAULT_OUTPUT = Path("data/ubc_ams_clubs.csv")
USER_AGENT = (
    "EventScraper AMS club contact scraper/1.0 (https://amsclubs.ca/all-clubs/)"
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
PAGE_NUMBER_PATTERN = re.compile(r"/pagenum/(\d+)/?")
INSTAGRAM_NON_ACCOUNT_PATHS = {
    "about",
    "accounts",
    "developer",
    "direct",
    "directory",
    "explore",
    "p",
    "reel",
    "reels",
    "stories",
    "tv",
}

log = logging.getLogger("scrape_ams_clubs")


@dataclass(frozen=True)
class Club:
    name: str
    page_url: str


@dataclass(frozen=True)
class ClubRow:
    club_name: str
    instagram_account_name: str
    email: str
    club_page_url: str


class HttpClient:
    """HTTP client with retry, timeout, and a delay between requests."""

    def __init__(self, delay: float = 0.25, timeout: float = 20.0) -> None:
        self.delay = delay
        self.timeout = timeout
        self._last_request_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    def get_soup(self, url: str) -> BeautifulSoup:
        if self._last_request_at is not None and self.delay:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

        response = self.session.get(url, timeout=self.timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")


def unique(values: Iterable[str]) -> list[str]:
    """Return non-empty values in encounter order without duplicates."""
    return list(dict.fromkeys(value for value in values if value))


def decode_cloudflare_email(encoded: str) -> str:
    """Decode Cloudflare's XOR-obfuscated email representation."""
    encoded = encoded.strip()
    if len(encoded) < 4 or len(encoded) % 2:
        return ""
    try:
        key = int(encoded[:2], 16)
        return "".join(
            chr(int(encoded[index : index + 2], 16) ^ key)
            for index in range(2, len(encoded), 2)
        )
    except (ValueError, UnicodeDecodeError):
        return ""


def extract_directory_entries(
    soup: BeautifulSoup, directory_url: str = DIRECTORY_URL
) -> list[Club]:
    clubs: list[Club] = []
    seen_urls: set[str] = set()

    for link in soup.select(".club-item a.club-item-link"):
        href = link.get("href")
        heading = link.select_one("h2")
        if not href or heading is None:
            continue
        url = urljoin(directory_url, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        clubs.append(Club(name=heading.get_text(" ", strip=True), page_url=url))

    return clubs


def directory_page_count(soup: BeautifulSoup) -> int:
    last_page = soup.select_one(".paging a.last-page")
    if last_page:
        match = PAGE_NUMBER_PATTERN.search(last_page.get("href", ""))
        if match:
            return max(1, int(match.group(1)))

    page_numbers = [
        int(match.group(1))
        for link in soup.select(".paging a[href]")
        if (match := PAGE_NUMBER_PATTERN.search(link.get("href", "")))
    ]
    return max(page_numbers, default=1)


def discover_clubs(
    client: HttpClient, directory_url: str = DIRECTORY_URL
) -> list[Club]:
    log.info("Fetching club directory: %s", directory_url)
    first_page = client.get_soup(directory_url)
    page_count = directory_page_count(first_page)
    clubs = extract_directory_entries(first_page, directory_url)

    for page_number in range(2, page_count + 1):
        page_url = urljoin(
            directory_url.rstrip("/") + "/",
            f"pagenum/{page_number}/",
        )
        log.info("Fetching directory page %d/%d", page_number, page_count)
        page = client.get_soup(page_url)
        clubs.extend(extract_directory_entries(page, directory_url))

    deduplicated: list[Club] = []
    seen_urls: set[str] = set()
    for club in clubs:
        if club.page_url not in seen_urls:
            seen_urls.add(club.page_url)
            deduplicated.append(club)

    if not deduplicated:
        raise RuntimeError("No clubs were found in the AMS directory")
    return deduplicated


def _instagram_handle(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if host not in {"instagram.com", "www.instagram.com"}:
        return ""

    parts = [unquote(part).strip() for part in parsed.path.split("/") if part.strip()]
    if parts and parts[0] == "_u":
        parts = parts[1:]
    if not parts or parts[0].lower() in INSTAGRAM_NON_ACCOUNT_PATHS:
        return ""
    return parts[0].lstrip("@")


def extract_instagram_accounts(soup: BeautifulSoup) -> list[str]:
    accounts: list[str] = []
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if "instagram.com" not in href.lower():
            continue
        handle = _instagram_handle(href)
        if handle:
            accounts.append(handle)

    return unique(accounts)


def extract_emails(soup: BeautifulSoup) -> list[str]:
    emails: list[str] = []

    for link in soup.select('a[href^="mailto:"]'):
        address = unquote(link.get("href", "")[7:]).split("?", 1)[0].strip()
        emails.extend(EMAIL_PATTERN.findall(address))

    for element in soup.select("[data-cfemail]"):
        decoded = decode_cloudflare_email(element.get("data-cfemail", ""))
        emails.extend(EMAIL_PATTERN.findall(decoded))

    for link in soup.select('a[href*="/cdn-cgi/l/email-protection#"]'):
        encoded = link.get("href", "").rsplit("#", 1)[-1]
        decoded = decode_cloudflare_email(encoded)
        emails.extend(EMAIL_PATTERN.findall(decoded))

    emails.extend(EMAIL_PATTERN.findall(soup.get_text(" ", strip=True)))
    return unique(address.lower() for address in emails)


def scrape_club(client: HttpClient, club: Club) -> ClubRow:
    try:
        soup = client.get_soup(club.page_url)
        instagram = "; ".join(extract_instagram_accounts(soup))
        email = "; ".join(extract_emails(soup))
    except requests.RequestException as exc:
        log.warning("Could not fetch %s: %s", club.page_url, exc)
        instagram = ""
        email = ""

    return ClubRow(
        club_name=club.name,
        instagram_account_name=instagram,
        email=email,
        club_page_url=club.page_url,
    )


def write_csv(rows: Iterable[ClubRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ("club_name", "instagram_account_name", "email", "club_page_url")
        )
        for row in rows:
            writer.writerow(
                (
                    row.club_name,
                    row.instagram_account_name,
                    row.email,
                    row.club_page_url,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape all UBC AMS clubs and their contact details."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Minimum delay between requests in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds (default: 20)",
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = HttpClient(delay=args.delay, timeout=args.timeout)

    try:
        clubs = discover_clubs(client)
    except (requests.RequestException, RuntimeError) as exc:
        log.error("Directory scrape failed: %s", exc)
        return 1

    log.info("Found %d clubs", len(clubs))
    rows: list[ClubRow] = []
    for index, club in enumerate(clubs, start=1):
        log.info("[%d/%d] Scraping %s", index, len(clubs), club.name)
        rows.append(scrape_club(client, club))

    write_csv(rows, args.output)
    log.info("Wrote %d rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
