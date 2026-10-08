from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import fitz
import httpx
from bs4 import BeautifulSoup

from data_collection.models import SourceDocument

BASE_URL = "https://www.airbank.cz"
PORADNA_URL = f"{BASE_URL}/poradna/"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
DOCUMENTS_URL = f"{BASE_URL}/dokumenty-ke-stazeni/?select=ke-smlouve"
ROBOTS_URL = f"{BASE_URL}/robots.txt"

USER_AGENT = "airbank-rag-portfolio-bot/1.0 (+educational portfolio project)"
ARTICLE_PREFIX = "/co-vas-nejvic-zajima/"

# These are useful for a customer-facing information assistant and intentionally
# exclude forms, historical versions and most legal/investment material.
CURATED_PDF_TITLES = (
    "Ceník",
    "Přehled úrokových sazeb",
    "Obchodní podmínky",
    "Podmínky platebního styku",
    "Podmínky doplňkových služeb",
    "Podmínky pro používání karet",
    "Podmínky pro používání hypotéky",
    "Podmínky pro používání úvěru",
    "Podmínky pro používání kontokorentu",
)

# Fallback discovery seeds if sitemap discovery is unavailable.
CATEGORY_SEEDS = (
    PORADNA_URL,
    f"{BASE_URL}/co-vas-nejvic-zajima/rubrika/bezpecnost/",
    f"{BASE_URL}/co-vas-nejvic-zajima/rubrika/bezny-a-sporici-ucet/",
    f"{BASE_URL}/co-vas-nejvic-zajima/rubrika/pujcky-a-kontokorent/",
    f"{BASE_URL}/co-vas-nejvic-zajima/rubrika/hypoteka/",
)

NOISE_TEXT = {
    "Zdá se, že máte vypnutý javascript. Zapněte si jej, aby vám tento web fungoval správně.",
    "Zeptejte se nás",
    "Zavoláme vám",
    "Napište nám",
    "Více informací",
}
STOP_SECTION_TEXT = {"Další témata"}


@dataclass(slots=True)
class CollectionStats:
    html_discovered: int = 0
    html_collected: int = 0
    pdf_discovered: int = 0
    pdf_collected: int = 0
    skipped: int = 0
    failed: int = 0


class AirBankCollector:
    """Collect a small, controlled corpus from Air Bank's public website."""

    def __init__(
        self,
        root_dir: Path,
        *,
        delay_seconds: float = 0.5,
        timeout_seconds: float = 30.0,
        respect_robots: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.raw_html_dir = root_dir / "data" / "raw" / "html"
        self.raw_pdf_dir = root_dir / "data" / "raw" / "pdf"
        self.processed_dir = root_dir / "data" / "processed"
        for directory in (self.raw_html_dir, self.raw_pdf_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.delay_seconds = max(delay_seconds, 0.0)
        self.respect_robots = respect_robots
        self._last_request_at = 0.0
        self._robots: RobotFileParser | None = None
        self._robots_checked = False
        self.client = httpx.Client(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "AirBankCollector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, url: str) -> httpx.Response:
        if self.respect_robots and not self._can_fetch(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        self._sleep_if_needed()
        response = self.client.get(url)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def _load_robots(self) -> None:
        if self._robots_checked:
            return
        self._robots_checked = True
        parser = RobotFileParser()
        parser.set_url(ROBOTS_URL)
        try:
            self._sleep_if_needed()
            response = self.client.get(ROBOTS_URL)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            parser.parse(response.text.splitlines())
            self._robots = parser
        except httpx.HTTPError:
            # If robots.txt cannot be fetched, continue conservatively with the
            # small rate-limited public corpus rather than failing the whole job.
            self._robots = None

    def _can_fetch(self, url: str) -> bool:
        self._load_robots()
        return self._robots is None or self._robots.can_fetch(USER_AGENT, url)

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        clean, _ = urldefrag(url)
        parsed = urlparse(clean)
        path = re.sub(r"/{2,}", "/", parsed.path)
        if path and not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
            path += "/"
        return parsed._replace(path=path, fragment="").geturl()

    @staticmethod
    def _is_airbank_url(url: str) -> bool:
        hostname = (urlparse(url).hostname or "").lower()
        return hostname in {"airbank.cz", "www.airbank.cz"}

    @classmethod
    def _is_article_url(cls, url: str) -> bool:
        if not cls._is_airbank_url(url):
            return False
        path = urlparse(url).path
        return (
            path.startswith(ARTICLE_PREFIX)
            and path != ARTICLE_PREFIX
            and "/rubrika/" not in path
        )

    def discover_article_urls(self, max_urls: int) -> list[str]:
        """Prefer sitemap discovery; fall back to a controlled link crawl."""
        try:
            urls = self._discover_from_sitemap(max_urls=max_urls)
            if urls:
                return urls[:max_urls]
        except (httpx.HTTPError, ET.ParseError, PermissionError):
            pass
        return self._discover_by_crawling(max_urls=max_urls)

    def _discover_from_sitemap(self, max_urls: int) -> list[str]:
        queue: deque[str] = deque([SITEMAP_URL])
        seen_sitemaps: set[str] = set()
        articles: set[str] = set()

        # Scan the sitemap fully before limiting. Taking the first N URLs created
        # an alphabetically biased corpus in the initial experiment.
        while queue:
            sitemap_url = queue.popleft()
            if sitemap_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sitemap_url)
            response = self._request(sitemap_url)
            root = ET.fromstring(response.content)
            root_name = root.tag.rsplit("}", 1)[-1]

            if root_name == "sitemapindex":
                for loc in root.iter():
                    if loc.tag.rsplit("}", 1)[-1] == "loc" and loc.text:
                        queue.append(loc.text.strip())
            elif root_name == "urlset":
                for loc in root.iter():
                    if loc.tag.rsplit("}", 1)[-1] != "loc" or not loc.text:
                        continue
                    url = self._canonicalize_url(loc.text.strip())
                    if self._is_article_url(url):
                        articles.add(url)

        # Stable hash ordering gives a deterministic spread across article slugs
        # instead of always selecting the same alphabetic prefix.
        ordered = sorted(
            articles,
            key=lambda url: hashlib.sha1(url.encode("utf-8")).hexdigest(),
        )
        return ordered[:max_urls]

    def _discover_by_crawling(self, max_urls: int) -> list[str]:
        queue: deque[str] = deque(CATEGORY_SEEDS)
        seen: set[str] = set()
        articles: set[str] = set()

        while queue and len(articles) < max_urls:
            current = self._canonicalize_url(queue.popleft())
            if current in seen:
                continue
            seen.add(current)
            try:
                response = self._request(current)
            except (httpx.HTTPError, PermissionError):
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.find_all("a", href=True):
                target = self._canonicalize_url(urljoin(str(response.url), anchor["href"]))
                if not self._is_airbank_url(target):
                    continue
                path = urlparse(target).path
                if self._is_article_url(target):
                    articles.add(target)
                elif target == PORADNA_URL or (
                    path.startswith(ARTICLE_PREFIX) and "/rubrika/" in path
                ):
                    if target not in seen:
                        queue.append(target)
                if len(articles) >= max_urls:
                    break

        return sorted(articles)

    @staticmethod
    def _extract_title(soup: BeautifulSoup, fallback_url: str) -> str:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
            if title:
                return title
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return str(og_title["content"]).strip()
        if soup.title and soup.title.string:
            return soup.title.string.strip().removesuffix(" | Air Bank")
        return urlparse(fallback_url).path.strip("/").split("/")[-1]

    @staticmethod
    def _clean_category(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s*Více informací\s*$", "", text, flags=re.IGNORECASE).strip()
        return text or "Poradna"

    @classmethod
    def _extract_category(cls, soup: BeautifulSoup) -> str:
        main = soup.find("main") or soup
        for anchor in main.find_all("a", href=True):
            href = str(anchor["href"])
            if f"{ARTICLE_PREFIX}rubrika/" in href:
                text = anchor.get_text(" ", strip=True)
                if text:
                    return cls._clean_category(text)
        return "Poradna"

    @staticmethod
    def _extract_html_text(soup: BeautifulSoup) -> str:
        for tag in soup.select("script, style, noscript, svg, form, button, nav, header, footer, aside"):
            tag.decompose()

        container = soup.find("main") or soup.find("article") or soup.body or soup
        blocks: list[str] = []
        previous = ""
        for element in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "th", "td"]):
            text = " ".join(element.stripped_strings)
            text = re.sub(r"\s+", " ", text).strip()
            if text in STOP_SECTION_TEXT:
                break
            if not text or text in NOISE_TEXT or text == previous:
                continue
            blocks.append(text)
            previous = text
        return "\n".join(blocks).strip()

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _safe_stem(url: str, fallback: str) -> str:
        slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1] or fallback
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-.")
        return slug[:120] or fallback

    @staticmethod
    def _document_id(url: str) -> str:
        slug = AirBankCollector._safe_stem(url, "document")
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        return f"{slug}-{digest}"

    def collect_html_document(self, url: str) -> SourceDocument | None:
        response = self._request(url)
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type:
            return None

        raw_bytes = response.content
        soup = BeautifulSoup(raw_bytes, "html.parser")
        text = self._extract_html_text(soup)
        if len(text) < 100:
            return None

        canonical_url = self._canonicalize_url(str(response.url))
        stem = self._safe_stem(canonical_url, "page")
        raw_path = self.raw_html_dir / f"{stem}.html"
        raw_path.write_bytes(raw_bytes)

        return SourceDocument(
            id=self._document_id(canonical_url),
            title=self._extract_title(soup, canonical_url),
            url=canonical_url,
            source_type="html",
            category=self._extract_category(soup),
            fetched_at=datetime.now(timezone.utc),
            text=text,
            content_sha256=self._sha256(raw_bytes),
            raw_path=str(raw_path.relative_to(self.root_dir)),
        )

    def discover_pdf_urls(self) -> list[tuple[str, str]]:
        response = self._request(DOCUMENTS_URL)
        soup = BeautifulSoup(response.text, "html.parser")
        wanted = set(CURATED_PDF_TITLES)
        found: dict[str, str] = {}

        for anchor in soup.find_all("a", href=True):
            title = " ".join(anchor.stripped_strings).strip()
            if title not in wanted or title in found:
                continue
            url = urljoin(str(response.url), anchor["href"])
            if self._is_airbank_url(url) and "/file-download/" in urlparse(url).path:
                found[title] = url

        # Stable order makes diffs/re-runs deterministic.
        return [(title, found[title]) for title in CURATED_PDF_TITLES if title in found]

    def collect_pdf_document(self, title: str, url: str) -> SourceDocument | None:
        response = self._request(url)
        raw_bytes = response.content
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not raw_bytes.startswith(b"%PDF"):
            return None

        stem = self._safe_stem(str(response.url), self._safe_stem(title, "document"))
        if not stem.lower().endswith(".pdf"):
            filename = f"{stem}.pdf"
        else:
            filename = stem
        raw_path = self.raw_pdf_dir / filename
        raw_path.write_bytes(raw_bytes)

        pdf = fitz.open(stream=raw_bytes, filetype="pdf")
        try:
            pages = [page.get_text("text", sort=True).strip() for page in pdf]
        finally:
            pdf.close()
        text = "\n\n".join(page for page in pages if page).strip()
        text = "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in text.replace("\u00a0", " ").splitlines()
        )
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 100:
            return None

        canonical_url = str(response.url)
        return SourceDocument(
            id=self._document_id(canonical_url),
            title=title,
            url=canonical_url,
            source_type="pdf",
            category="Official documents",
            fetched_at=datetime.now(timezone.utc),
            text=text,
            content_sha256=self._sha256(raw_bytes),
            raw_path=str(raw_path.relative_to(self.root_dir)),
        )

    def collect(
        self,
        *,
        max_html: int = 200,
        include_html: bool = True,
        include_pdfs: bool = True,
    ) -> tuple[list[SourceDocument], CollectionStats]:
        stats = CollectionStats()
        documents: list[SourceDocument] = []

        if include_html:
            html_urls = self.discover_article_urls(max_urls=max_html)
            stats.html_discovered = len(html_urls)
            for index, url in enumerate(html_urls, start=1):
                print(f"[html {index}/{len(html_urls)}] {url}")
                try:
                    document = self.collect_html_document(url)
                    if document is None:
                        stats.skipped += 1
                    else:
                        documents.append(document)
                        stats.html_collected += 1
                except (httpx.HTTPError, PermissionError, OSError) as exc:
                    stats.failed += 1
                    print(f"  ! failed: {exc}")

        if include_pdfs:
            try:
                pdf_urls = self.discover_pdf_urls()
            except (httpx.HTTPError, PermissionError) as exc:
                pdf_urls = []
                stats.failed += 1
                print(f"! PDF discovery failed: {exc}")
            stats.pdf_discovered = len(pdf_urls)
            for index, (title, url) in enumerate(pdf_urls, start=1):
                print(f"[pdf {index}/{len(pdf_urls)}] {title}: {url}")
                try:
                    document = self.collect_pdf_document(title, url)
                    if document is None:
                        stats.skipped += 1
                    else:
                        documents.append(document)
                        stats.pdf_collected += 1
                except (httpx.HTTPError, PermissionError, OSError, RuntimeError) as exc:
                    stats.failed += 1
                    print(f"  ! failed: {exc}")

        # URL-level de-duplication, preserving first occurrence.
        unique: dict[str, SourceDocument] = {}
        for document in documents:
            unique.setdefault(str(document.url), document)
        return list(unique.values()), stats

    def write_documents(
        self,
        documents: Iterable[SourceDocument],
        *,
        output_name: str = "documents.jsonl",
    ) -> Path:
        output_path = self.processed_dir / output_name
        with output_path.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(document.model_dump_json() + "\n")
        return output_path

    def write_summary(
        self,
        documents: list[SourceDocument],
        stats: CollectionStats,
        *,
        output_name: str = "collection_summary.json",
    ) -> Path:
        output_path = self.processed_dir / output_name
        fingerprint_rows = sorted(
            (document.id, str(document.url), document.content_sha256)
            for document in documents
        )
        fingerprint_payload = json.dumps(
            fingerprint_rows, ensure_ascii=False, separators=(",", ":")
        )
        content_fingerprint = hashlib.sha256(
            fingerprint_payload.encode("utf-8")
        ).hexdigest()

        summary = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "content_fingerprint": content_fingerprint,
            "documents": len(documents),
            "html_documents": sum(d.source_type == "html" for d in documents),
            "pdf_documents": sum(d.source_type == "pdf" for d in documents),
            "stats": {
                "html_discovered": stats.html_discovered,
                "html_collected": stats.html_collected,
                "pdf_discovered": stats.pdf_discovered,
                "pdf_collected": stats.pdf_collected,
                "skipped": stats.skipped,
                "failed": stats.failed,
            },
            "categories": sorted({d.category for d in documents}),
        }
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path
