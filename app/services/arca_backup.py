import base64
import json
import logging
import mimetypes
import os
import re
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlencode, urljoin, urlparse, parse_qs, urlunparse

import requests
from bs4 import BeautifulSoup

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".webm", ".ogg", ".mov", ".mkv")
BASE_URL = "https://arca.live/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    return raw if raw is not None else default


def _load_cookies_to_session(session: requests.Session, state_path: str) -> None:
    """storage_state.json 쿠키를 requests 세션에 동기화한다."""
    if not state_path or not os.path.exists(state_path):
        return
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        cookies = data.get("cookies", []) if isinstance(data, dict) else []
        loaded = 0
        for c in cookies:
            try:
                session.cookies.set(
                    c.get("name"),
                    c.get("value"),
                    domain=c.get("domain"),
                    path=c.get("path"),
                )
                loaded += 1
            except Exception:
                continue
        logger.info("Playwright 쿠키 %s개를 requests 세션에 적용했습니다.", loaded)
    except Exception as exc:
        logger.warning("쿠키 로드 실패: %s", exc)


def _run_playwright_job(job):
    """
    Playwright Sync API는 실행 중인 asyncio 루프 안에서 직접 호출하면 경고/예외가 납니다.
    루프가 돌고 있다면 별도 스레드에서 실행하여 충돌을 피합니다.
    """
    try:
        loop = asyncio.get_event_loop()
        running = loop.is_running()
    except Exception:
        running = False

    if not running:
        return job()

    result_holder = {}
    error_holder = {}

    def worker():
        try:
            result_holder["value"] = job()
        except Exception as exc:
            error_holder["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("value")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name or "").strip()
    return cleaned[:120] if len(cleaned) > 120 else cleaned


def _get_value(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key)
    if hasattr(item, key):
        return getattr(item, key)
    getter = getattr(item, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return None


def _extract_article_id_from_url(url: Any, channel: str) -> Optional[int]:
    if not isinstance(url, str) or not url:
        return None
    try:
        full = urljoin(BASE_URL, url)
        path = urlparse(full).path
    except Exception:
        return None

    if channel:
        match = re.search(rf"/b/{re.escape(channel)}/(\d+)", path)
        if match:
            return int(match.group(1))

    match = re.search(r"/b/[^/]+/(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def _has_articleish_fields(item: Any) -> bool:
    for key in ("title", "subject", "url", "href"):
        value = _get_value(item, key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def pick_article_id(item: Any, channel: str) -> Optional[int]:
    for url_key in ("url", "href", "link"):
        candidate = _extract_article_id_from_url(_get_value(item, url_key), channel)
        if candidate:
            return candidate

    for key in ("id", "article_id", "aid", "no"):
        value = _get_value(item, key)
        if key == "no" and not _has_articleish_fields(item):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            parsed = int(value)
            if parsed > 0:
                return parsed
    return None


def _iter_children(node: Any) -> Iterable[Any]:
    if isinstance(node, dict):
        return node.values()
    if isinstance(node, (list, tuple, set)):
        return node
    if hasattr(node, "__dict__"):
        return vars(node).values()
    return ()


def extract_article_ids(listing: Any, channel: str) -> List[int]:
    ids: List[int] = []
    seen: Set[int] = set()

    def visit(node: Any) -> None:
        if node is None:
            return

        if isinstance(node, (dict, list, tuple, set)) or hasattr(node, "__dict__"):
            marker = id(node)
            if marker in seen:
                return
            seen.add(marker)

        if isinstance(node, str):
            found = _extract_article_id_from_url(node, channel)
            if found:
                ids.append(found)
            for match in re.finditer(rf"/b/{re.escape(channel)}/(\d+)", node):
                ids.append(int(match.group(1)))
        else:
            found = pick_article_id(node, channel)
            if found:
                ids.append(found)

        for child in _iter_children(node):
            visit(child)

    visit(listing)
    return list(dict.fromkeys(ids))


def extract_image_urls(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html or "", "lxml")
    urls: List[str] = []

    for img in soup.find_all("img"):
        src = img.get("src")
        if not src or src.startswith("data:"):
            src = img.get("data-src") or img.get("data-original")
        if not src:
            continue

        full_url = urljoin(base_url, src)

        # 무거운 원본(type=orig) 파라미터는 제거하여 경량 이미지로 요청
        try:
            parsed = urlparse(full_url)
            qs = parse_qs(parsed.query)
            if "type" in qs and "orig" in qs["type"]:
                qs.pop("type", None)
                new_query = urlencode(qs, doseq=True)
                full_url = urlunparse(parsed._replace(query=new_query))
        except Exception:
            pass

        urls.append(full_url)

    deduped: List[str] = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def extract_video_urls(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html or "", "lxml")
    urls: List[str] = []

    for video in soup.find_all("video"):
        src = (
            video.get("data-originalurl")
            or video.get("src")
            or video.get("data-src")
            or video.get("data-original")
        )
        if src:
            urls.append(urljoin(base_url, src))
        for source in video.find_all("source"):
            s = source.get("src") or source.get("data-src") or source.get("data-original")
            if s:
                urls.append(urljoin(base_url, s))

    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        path = urlparse(full).path.lower()
        if any(path.endswith(ext) for ext in VIDEO_EXTS):
            urls.append(full)

    deduped: List[str] = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def download_as_data_url(url: str, referer: str, session: requests.Session) -> str:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": referer,
    }
    response = session.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip() or None
    if not content_type:
        guessed, _ = mimetypes.guess_type(url)
        content_type = guessed or "application/octet-stream"
    data = b"".join(response.iter_content(chunk_size=1024 * 256))
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def download_file(url: str, referer: str, session: requests.Session, dest: Path) -> Path:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": referer,
    }
    response = session.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 512):
            if not chunk:
                continue
            handle.write(chunk)
    return dest


def build_channel_url(channel: str, category: str, page: int) -> str:
    base = urljoin(BASE_URL, f"b/{channel}")
    params = {"p": page}
    if category:
        params["category"] = category
    return f"{base}?{urlencode(params)}"


def build_article_url(channel: str, aid: int) -> str:
    return urljoin(BASE_URL, f"b/{channel}/{aid}")


def extract_last_page_number(soup: BeautifulSoup, listing_url: str, channel: str) -> Optional[int]:
    max_page = None
    base_path = f"/b/{channel}".lower()

    for anchor in soup.find_all("a"):
        href = anchor.get("href") or ""
        if not href:
            continue
        try:
            full_href = urljoin(listing_url, href)
            parsed = urlparse(full_href)
        except Exception:
            continue

        path = (parsed.path or "").lower()
        if base_path not in path:
            continue

        candidates: List[str] = []
        data_page = anchor.get("data-page")
        if data_page is not None:
            candidates.append(str(data_page))

        qs = parse_qs(parsed.query)
        candidates.extend(qs.get("p") or [])

        for cand in candidates:
            try:
                page_num = int(str(cand))
            except (TypeError, ValueError):
                continue
            if page_num < 1:
                continue
            if max_page is None or page_num > max_page:
                max_page = page_num

    return max_page


class PlaywrightListingClient:
    def __init__(self, headless: bool = True, timeout_ms: int = 30_000, storage_state: str = ""):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.storage_state = storage_state
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_started(self) -> None:  # pragma: no cover - environment dependent
        if self._page is not None:
            return

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "Playwright가 설치되어 있지 않습니다. `pip install playwright` 후 "
                "`python -m playwright install chromium`을 실행하세요."
            ) from exc

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._context = self._browser.new_context(
                user_agent=DEFAULT_USER_AGENT,
                locale="ko-KR",
                extra_http_headers={
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                },
                storage_state=self.storage_state or None,
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)
        except PlaywrightError as exc:
            self.close()
            raise RuntimeError(
                "Playwright 브라우저 실행에 실패했습니다: "
                f"{exc}. `python -m playwright install chromium` 실행 여부를 확인하세요."
            ) from exc

    def close(self) -> None:  # pragma: no cover - environment dependent
        try:
            if self._page is not None:
                self._page.close()
        finally:
            self._page = None

        try:
            if self._context is not None:
                self._context.close()
        finally:
            self._context = None

        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._browser = None

        try:
            if self._playwright is not None:
                self._playwright.stop()
        finally:
            self._playwright = None

    def collect_article_ids(
        self, channel: str, category: str, page: int, debug_dir: Optional[Path] = None
    ) -> Tuple[List[int], Optional[int]]:
        self._ensure_started()

        if self._page is None:
            return []

        url = build_channel_url(channel=channel, category=category, page=page)
        request_events: List[Dict[str, Any]] = []
        response_events: List[Dict[str, Any]] = []
        error_events: List[Dict[str, Any]] = []
        json_ids: Set[int] = set()
        last_page_detected: Optional[int] = None

        def on_request(req):  # pragma: no cover - network-dependent
            try:
                request_events.append(
                    {
                        "url": req.url,
                        "method": req.method,
                        "resource_type": req.resource_type,
                    }
                )
            except Exception:
                return

        def on_response(res):  # pragma: no cover - network-dependent
            try:
                response_events.append(
                    {
                        "url": res.url,
                        "status": res.status,
                        "resource_type": res.request.resource_type,
                    }
                )
            except Exception:
                return

            try:
                if res.status >= 400:
                    return
                headers = res.headers or {}
                content_type = (headers.get("content-type") or "").lower()
                if "application/json" not in content_type:
                    return
                payload = res.json()
                for aid in extract_article_ids(payload, channel):
                    json_ids.add(aid)
            except Exception:
                return

        def on_request_failed(req):  # pragma: no cover - network-dependent
            try:
                error_events.append(
                    {
                        "url": req.url,
                        "method": req.method,
                        "resource_type": req.resource_type,
                        "failure": (req.failure or {}).get("errorText"),
                    }
                )
            except Exception:
                return

        self._page.on("request", on_request)
        self._page.on("response", on_response)
        self._page.on("requestfailed", on_request_failed)

        try:
            response = self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            status = response.status if response is not None else None
            if status is not None and status >= 400:
                status_text = getattr(response, "status_text", "") if response is not None else ""
                raise RuntimeError(f"{status} {status_text} for url: {url}")

            try:
                self._page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass

            pattern = re.compile(rf"^/b/{re.escape(channel)}/(\d+)")
            ids: List[int] = []
            for attempt in range(1, 61):
                html = self._page.content()
                soup = BeautifulSoup(html, "lxml")
                ids = []
                table = soup.select_one("div.list-table.table")
                if table:
                    for a in table.select("a[href^='/b/']"):
                        # 공지/필터된 공지 건너뜀
                        classes = a.get("class") or []
                        if any("notice" in c for c in classes):
                            continue
                        href = a.get("href") or ""
                        # 쿼리 파라미터 제거 후 경로로만 매칭
                        parsed_path = urlparse(href).path or href
                        m = pattern.match(parsed_path)
                        if m:
                            ids.append(int(m.group(1)))
                ids = list(dict.fromkeys(ids))

                if last_page_detected is None:
                    try:
                        last_page_detected = extract_last_page_number(soup, url, channel)
                    except Exception:
                        last_page_detected = None

                if ids:
                    return ids, last_page_detected
                if json_ids:
                    return list(sorted(json_ids)), last_page_detected
                try:
                    self._page.evaluate("window.scrollBy(0, 1200)")
                except Exception:
                    pass
                self._page.wait_for_timeout(500)
        finally:
            try:
                self._page.off("request", on_request)
                self._page.off("response", on_response)
                self._page.off("requestfailed", on_request_failed)
            except Exception:
                pass

        title = ""
        snippet = ""
        body_excerpt = ""
        try:
            title = self._page.title() or ""
        except Exception:
            title = ""

        try:
            snippet = (self._page.inner_text("body") or "").strip().replace("\n", " ")
            snippet = snippet[:160]
        except Exception:
            snippet = ""

        try:
            body_excerpt = (self._page.content() or "")[:5000]
        except Exception:
            body_excerpt = ""

        debug_paths: Dict[str, str] = {}
        if debug_dir is not None:
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                suffix = safe_filename(category) or "all"
                base_name = f"listing_{channel}_{suffix}_p{page}"
                html_path = debug_dir / f"{base_name}.html"
                shot_path = debug_dir / f"{base_name}.png"
                net_path = debug_dir / f"{base_name}.network.json"

                html_full = ""
                try:
                    html_full = self._page.content() or ""
                except Exception:
                    html_full = ""
                with open(html_path, "w", encoding="utf-8") as handle:
                    handle.write(html_full)

                try:
                    self._page.screenshot(path=str(shot_path), full_page=True)
                except Exception:
                    pass

                with open(net_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "url": url,
                            "final_url": getattr(self._page, "url", ""),
                            "title": title,
                            "requests": request_events[-200:],
                            "responses": response_events[-200:],
                            "request_failed": error_events[-200:],
                        },
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )

                debug_paths = {
                    "html": str(html_path),
                    "screenshot": str(shot_path),
                    "network": str(net_path),
                }
            except Exception:
                debug_paths = {}

        logger.warning(
            "[playwright] 글 링크 추출 실패 url=%s title=%s snippet=%r body_len=%s debug=%s",
            url,
            title,
            snippet,
            len(body_excerpt),
            debug_paths or None,
        )

        if snippet:
            raise RuntimeError(f"목록 페이지에서 글 링크를 찾지 못했습니다: {url} (title={title}, body={snippet!r})")
        raise RuntimeError(f"목록 페이지에서 글 링크를 찾지 못했습니다: {url} (title={title}, body_len={len(body_excerpt)})")


class PlaywrightArticleClient:
    def __init__(self, headless: bool = True, timeout_ms: int = 30_000, storage_state: str = ""):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.storage_state = storage_state
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_started(self) -> None:  # pragma: no cover - environment dependent
        if self._page is not None:
            return
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "Playwright가 설치되어 있지 않습니다. `pip install playwright` 후 "
                "`python -m playwright install chromium`을 실행하세요."
            ) from exc

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._context = self._browser.new_context(
                user_agent=DEFAULT_USER_AGENT,
                locale="ko-KR",
                extra_http_headers={
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                },
                storage_state=self.storage_state or None,
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)
        except PlaywrightError as exc:
            self.close()
            raise RuntimeError(
                "Playwright 브라우저 실행에 실패했습니다: "
                f"{exc}. `python -m playwright install chromium` 실행 여부를 확인하세요."
            ) from exc

    def close(self) -> None:  # pragma: no cover - environment dependent
        try:
            if self._page is not None:
                self._page.close()
        finally:
            self._page = None

        try:
            if self._context is not None:
                self._context.close()
        finally:
            self._context = None

        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._browser = None

        try:
            if self._playwright is not None:
                self._playwright.stop()
        finally:
            self._playwright = None

    def fetch_article(self, channel: str, aid: int) -> Dict[str, Any]:
        self._ensure_started()
        if self._page is None:
            raise RuntimeError("Playwright 페이지 초기화에 실패했습니다.")

        url = build_article_url(channel, aid)
        response = self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        status = response.status if response is not None else None
        if status is not None and status >= 400:
            status_text = getattr(response, "status_text", "") if response is not None else ""
            raise RuntimeError(f"{status} {status_text} for url: {url}")

        try:
            self._page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass

        try:
            self._page.evaluate("window.scrollBy(0, 2000)")
        except Exception:
            pass

        html = self._page.content()
        title = ""
        try:
            title = self._page.title() or ""
        except Exception:
            title = ""

        # Detect permission/blocked page
        body_text = ""
        try:
            body_text = (self._page.inner_text("body") or "").strip()
        except Exception:
            body_text = ""
        # Extract article HTML - project 요구사항: article-wrapper/본문만 사용
        page_soup = BeautifulSoup(html, "lxml")
        is_notice = False
        try:
            slug_match = re.search(r'"articleBoardSlug":"([^"]+)"', html)
            if slug_match and slug_match.group(1) == "notice":
                is_notice = True
        except Exception:
            pass

        selectors = [
            "div.article-body",        # 본문 영역
            "div.article-wrapper",     # 헤더+본문 묶음
            "article.article-wrapper", # 일부 페이지 변형 대비
        ]
        found = None
        for sel in selectors:
            found = page_soup.select_one(sel)
            if found:
                break
        badge = page_soup.select_one(".category-badge")
        badge_text = badge.get_text(strip=True) if badge else ""
        if badge_text == "공지":
            is_notice = True
        content_html = str(found) if found else html
        soup_snippet = found.get_text(strip=True)[:200] if found else ""
        matched_selector = (
            selectors[selectors.index(sel)] if found else "full_page"
        )

        author = ""
        try:
            author_elem = page_soup.select_one(".member-info .user-info a") or page_soup.select_one(
                ".member-info .user-info"
            )
            if author_elem:
                author = author_elem.get_text(strip=True)
        except Exception:
            author = ""

        result = {
            "title": title or f"post_{aid}",
            "content": content_html,
            "url": self._page.url or url,
            "raw_html": html,
            "matched_selector": matched_selector,
            "snippet": soup_snippet,
            "is_notice": is_notice,
            "author": author,
        }
        return result


@dataclass
class BackupConfig:
    channel: str
    category: str
    start_page: int
    end_page: Optional[int]
    out_dir: Path
    sleep: float


@dataclass
class BackupResult:
    manifest: List[Dict]
    errors: List[str]
    out_dir: Path
    pages_processed: int
    posts_saved: int
    images_downloaded: int
    videos_downloaded: int
    duration: float
    started_at: float
    finished_at: float
    end_page: Optional[int]
    auto_last_page: bool


def backup_channel(config: BackupConfig, progress_callback=None, stop_event=None) -> BackupResult:
    def log_progress(msg: str, level=logging.INFO):
        if level == logging.ERROR:
            logger.error(msg)
        elif level == logging.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)
        
        if progress_callback:
            progress_callback(msg)

    started_at = time.time()
    session = requests.Session()
    storage_state_path = env_str("PLAYWRIGHT_STORAGE_STATE", "")
    _load_cookies_to_session(session, storage_state_path)
    errors: List[str] = []
    posts_saved = 0
    images_downloaded = 0
    videos_downloaded = 0
    pages_processed = 0
    page_listings: List[Dict[str, Any]] = []
    all_ids: List[int] = []
    auto_last_page = config.end_page is None
    effective_end_page = config.end_page if config.end_page is not None else config.start_page
    resolved_end_page: Optional[int] = config.end_page
    last_listing_page = config.start_page - 1
    end_page_label = effective_end_page if not auto_last_page else "자동 탐색"

    def should_stop() -> bool:
        return bool(stop_event) and stop_event.is_set()

    config.out_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"백업 시작: {config.channel} (페이지 {config.start_page}~{end_page_label})")
    if auto_last_page:
        log_progress("마지막 페이지가 비어 있어 자동으로 탐색합니다.")

    # 1) Listing 수집 단계
    page = config.start_page
    consecutive_no_results = 0

    # 기존 목록 로드 (증분 백업용)
    existing_ids = set()
    listing_path = config.out_dir / "_listing.json"
    old_listings = []
    if listing_path.exists():
        try:
            with open(listing_path, "r", encoding="utf-8") as f:
                old_listings = json.load(f)
                if isinstance(old_listings, list):
                    for entry in old_listings:
                        existing_ids.update(entry.get("ids", []))
            if existing_ids:
                log_progress(f"기존 목록에서 {len(existing_ids)}개의 게시글 ID를 로드했습니다. 증분 백업을 시도합니다.")
        except Exception as exc:
            logger.warning("기존 _listing.json 로드 실패: %s", exc)

    while True:
        if not auto_last_page and page > effective_end_page:
            break
        if auto_last_page and resolved_end_page is not None and page > resolved_end_page:
            break
        if should_stop():
            log_progress("중지 요청을 감지했습니다. 목록 수집을 중단합니다.")
            break
        last_listing_page = page
        pages_processed += 1
        log_progress(f"목록 수집 중: {page}페이지...")
        playwright_error: Optional[Exception] = None
        ids: List[int] = []
        detected_last_page: Optional[int] = None

        def collect_ids(target_category: str) -> Tuple[List[int], Optional[int]]:
            def job():
                client = PlaywrightListingClient(
                    headless=env_bool("PLAYWRIGHT_HEADLESS", True),
                    storage_state=env_str("PLAYWRIGHT_STORAGE_STATE", ""),
                )
                try:
                    return client.collect_article_ids(
                        channel=config.channel,
                        category=target_category,
                        page=page,
                        debug_dir=config.out_dir / "_debug" / "listing",
                    )
                finally:
                    client.close()
            return _run_playwright_job(job)

        try:
            ids, detected_last_page = collect_ids(config.category)
        except Exception as exc:  # pragma: no cover - network-dependent
            playwright_error = exc
            logger.warning("[page %s] playwright listing 실패: %s", page, playwright_error)

        # 증분 백업 체크
        found_existing = False
        if existing_ids and ids:
            new_ids_in_page = [i for i in ids if i not in existing_ids]
            if not new_ids_in_page:
                log_progress("이미 백업된 게시글 지점에 도달했습니다. 목록 수집을 종료합니다.")
                break
            if len(new_ids_in_page) < len(ids):
                log_progress(f"기존 게시글을 발견했습니다. {page}페이지에서 목록 수집을 종료합니다.")
                ids = new_ids_in_page
                found_existing = True

        if auto_last_page and detected_last_page:
            if resolved_end_page is None or detected_last_page > resolved_end_page:
                resolved_end_page = detected_last_page
                if detected_last_page > effective_end_page:
                    effective_end_page = detected_last_page
                log_progress(f"마지막 페이지 감지: {detected_last_page}페이지 (범위 확장)")

        listing_error: Optional[Exception] = None
        detail = ""
        if not ids:
            if playwright_error is not None or not auto_last_page:
                listing_error = RuntimeError("목록에서 글 ID를 발견할 수 없습니다.")

        if not ids and (playwright_error is not None or listing_error is not None):
            detail_parts = []
            if playwright_error is not None:
                detail_parts.append(f"playwright: {playwright_error}")
            if listing_error is not None:
                detail_parts.append(f"arcalive: {listing_error}")
            detail = " | ".join(detail_parts) if detail_parts else "unknown"
            logger.warning(
                "[page %s] 글 ID 추출 실패 channel=%s category=%s detail=%s",
                page,
                config.channel,
                config.category or "(all)",
                detail,
            )
            errors.append(f"[page {page}] 글 ID를 찾지 못했습니다(구조 변경/차단 가능): {detail}")

        page_listings.append(
            {
                "page": page,
                "category": config.category,
                "ids": ids,
                "error": detail,
            }
        )

        if ids:
            all_ids.extend(ids)
            consecutive_no_results = 0
        else:
            consecutive_no_results += 1

        if found_existing:
            break

        if auto_last_page and not ids and not detail:
            stop_page = page - 1 if page > config.start_page else page
            resolved_end_page = stop_page
            log_progress(f"게시글이 없는 페이지를 확인했습니다. {stop_page}페이지에서 목록 수집을 종료합니다.")
            break

        if auto_last_page and not ids and detail and consecutive_no_results >= 3:
            resolved_end_page = page - 1 if page > config.start_page else page
            log_progress("3개 연속 페이지에서 글을 찾지 못해 목록 수집을 중단합니다.")
            break

        page += 1
        time.sleep(config.sleep)

    if resolved_end_page is None:
        resolved_end_page = last_listing_page if last_listing_page >= config.start_page else config.end_page

    # listing 결과 저장
    final_listings = page_listings + old_listings
    with open(config.out_dir / "_listing.json", "w", encoding="utf-8") as handle:
        json.dump(final_listings, handle, ensure_ascii=False, indent=2)

    # 2) 본문 크롤링 단계
    all_ids = list(dict.fromkeys(all_ids))
    total_ids = len(all_ids)
    log_progress(f"본문 크롤링 시작 (총 {total_ids}개)...")

    # 기존 manifest 로드 (중복 스킵용)
    existing_posts = {}
    manifest_path = config.out_dir / "_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                old_data = json.load(handle)
                if isinstance(old_data, list):
                    for item in old_data:
                        aid_val = item.get("id")
                        if aid_val:
                            existing_posts[aid_val] = item
        except Exception as exc:
            log_progress(f"기존 manifest 로드 실패: {exc}", level=logging.WARNING)

    # 매니페스트에 없더라도 실제 폴더가 존재하는지 스캔 (Fallback)
    for p in config.out_dir.iterdir():
        if not p.is_dir():
            continue
        # 폴더명이 {aid}_{title} 형식이므로 aid 추출 시도
        name_parts = p.name.split("_", 1)
        if not name_parts[0].isdigit():
            continue
        
        aid_val = int(name_parts[0])
        if aid_val in existing_posts:
            continue
            
        # index.html과 post.json이 모두 있어야 완료된 것으로 간주
        if (p / "index.html").exists() and (p / "post.json").exists():
            try:
                with open(p / "post.json", "r", encoding="utf-8") as h:
                    post_data = json.load(h)
                
                # 실제 폴더 내 이미지 개수 확인
                img_count = 0
                for f in p.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        img_count += 1

                existing_posts[aid_val] = {
                    "id": aid_val,
                    "title": post_data.get("title", p.name),
                    "url": post_data.get("url", ""),
                    "dir": p.name,
                    "images": img_count,
                    "downloaded_images": img_count,
                }
            except Exception:
                continue

    if should_stop():
        log_progress("중지 요청을 감지했습니다. 본문 크롤링을 건너뜁니다.")

    # 누적 매니페스트를 위해 기존 항목으로 시작
    manifest_map = {item["id"]: item for item in existing_posts.values() if "id" in item}

    for idx, aid in enumerate(all_ids, 1):
        if should_stop():
            log_progress("중지 요청을 감지했습니다. 남은 게시물 처리를 중단합니다.")
            break
        if aid in manifest_map:
            item = manifest_map[aid]
            # 실제 폴더와 최종 결과물(index.html)이 존재하는지 한 번 더 확인
            target_dir = config.out_dir / item.get("dir", "")
            if (target_dir / "index.html").exists():
                log_progress(f"[{idx}/{total_ids}] [{aid}] 이미 완료된 게시물입니다. 스킵합니다.")
                continue
            else:
                log_progress(f"[{idx}/{total_ids}] [{aid}] 매니페스트에는 있으나 결과물 파일이 없습니다. 다시 크롤링합니다.")

        log_progress(f"[{idx}/{total_ids}] [{aid}] 크롤링 중...")
        fetch_ms = img_ms = vid_ms = poster_ms = rewrite_ms = 0.0
        post_start = time.perf_counter()
        try:
            fetch_start = time.perf_counter()
            def fetch_job():
                client = PlaywrightArticleClient(
                    headless=env_bool("PLAYWRIGHT_HEADLESS", True),
                    storage_state=env_str("PLAYWRIGHT_STORAGE_STATE", ""),
                )
                try:
                    return client.fetch_article(config.channel, aid)
                finally:
                    client.close()
            article = _run_playwright_job(fetch_job)
            fetch_ms = (time.perf_counter() - fetch_start) * 1000
        except Exception as exc:  # pragma: no cover - depends on upstream API
            err_msg = f"[{aid}] 글 조회 실패: {exc}"
            log_progress(err_msg, level=logging.ERROR)
            errors.append(err_msg)
            time.sleep(config.sleep)
            continue

        if article.get("is_notice"):
            log_progress(f"[{aid}] 공지 글이라 스킵했습니다.")
            errors.append(f"[{aid}] 공지 글이라 스킵했습니다.")
            time.sleep(config.sleep)
            continue

        title = article.get("title") or f"post_{aid}"
        log_progress(f"[{aid}] 저장 중: {title[:30]}...")
        content_html = article.get("content") or ""
        post_url = article.get("url") or build_article_url(config.channel, aid)
        author = article.get("author") or ""

        post_dir = config.out_dir / f"{aid}_{safe_filename(title)}"
        post_dir.mkdir(parents=True, exist_ok=True)

        with open(post_dir / "post.json", "w", encoding="utf-8") as handle:
            json.dump(article, handle, ensure_ascii=False, indent=2)

        image_urls = extract_image_urls(content_html, base_url=post_url)
        video_urls = extract_video_urls(content_html, base_url=post_url)
        url_to_local: Dict[str, str] = {}
        video_to_local: Dict[str, str] = {}
        video_fallback_image: Dict[str, bool] = {}
        video_poster_map: Dict[str, str] = {}
        poster_to_local: Dict[str, str] = {}

        if content_html:
            soup_media = BeautifulSoup(content_html, "lxml")
            for video in soup_media.find_all("video"):
                v_src = (
                    video.get("data-originalurl")
                    or video.get("src")
                    or video.get("data-src")
                    or video.get("data-original")
                )
                if not v_src:
                    continue
                v_full = urljoin(post_url, v_src)
                poster = video.get("poster")
                if poster:
                    video_poster_map[v_full] = urljoin(post_url, poster)

        if image_urls and not should_stop():
            img_start = time.perf_counter()
            max_workers = 1

            def download_one(target_url: str):
                base_delay = 0.5
                for attempt in range(3):
                    try:
                        data_url = download_as_data_url(
                            target_url,
                            referer=post_url,
                            session=session,
                        )
                        return target_url, data_url
                    except requests.HTTPError as exc:
                        status = exc.response.status_code if exc.response is not None else None
                        if status == 429:
                            wait = base_delay * (2 ** attempt)
                            time.sleep(wait)
                            continue
                        raise
                    except Exception:
                        if attempt < 2:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        raise
                # 마지막 실패 시 예외 전파
                data_url = download_as_data_url(target_url, referer=post_url, session=session)
                return target_url, data_url

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(download_one, url): url for url in image_urls}
                for future in futures:
                    if should_stop():
                        log_progress("중지 요청을 감지했습니다. 이미지 다운로드를 중단합니다.")
                        break
                    try:
                        url, data_url = future.result()
                        url_to_local[url] = data_url
                        images_downloaded += 1
                    except Exception as exc:  # pragma: no cover - network-dependent
                        err_msg = f"[{aid}] 이미지 다운로드 실패: {futures[future]} ({exc})"
                        errors.append(err_msg)
                        log_progress(err_msg, level=logging.WARNING)
                    # per-download throttle to avoid bursts
                    time.sleep(0.5)
            img_ms = (time.perf_counter() - img_start) * 1000

        if video_urls and not should_stop():
            vid_start = time.perf_counter()
            max_workers = min(3, max(1, len(video_urls)))
            videos_dir = post_dir / "videos"

            def download_video(target_url: str):
                headers = {
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Referer": post_url,
                }
                base_delay = 0.5
                temp_path = None
                for attempt in range(3):
                    try:
                        response = session.get(target_url, headers=headers, stream=True, timeout=60)
                        response.raise_for_status()
                        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()

                        parsed = urlparse(target_url)
                        name = Path(parsed.path).name or "video"
                        safe_name = safe_filename(name) or "video"

                        guessed_ext = None
                        if content_type:
                            ext_from_type = mimetypes.guess_extension(content_type)
                            if ext_from_type:
                                guessed_ext = ext_from_type
                        if not guessed_ext:
                            guessed_ext = Path(safe_name).suffix

                        is_video_type = content_type.startswith("video/") or any(
                            (guessed_ext or "").lower().endswith(ext) for ext in VIDEO_EXTS
                        )

                        if not is_video_type:
                            # Treat as image/gif/webp fallback
                            if not guessed_ext:
                                guessed_ext = ".gif"
                            safe_name = Path(safe_name).stem + guessed_ext
                        else:
                            if not any(safe_name.lower().endswith(ext) for ext in VIDEO_EXTS):
                                safe_name = f"{Path(safe_name).stem}.mp4"

                        dest = videos_dir / safe_name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        temp_path = dest.with_suffix(dest.suffix + ".part")
                        with open(temp_path, "wb") as handle:
                            for chunk in response.iter_content(chunk_size=1024 * 512):
                                if chunk:
                                    handle.write(chunk)
                        # atomic rename
                        temp_path.replace(dest)
                        return target_url, dest.name, not is_video_type
                    except requests.HTTPError as exc:
                        status = exc.response.status_code if exc.response is not None else None
                        if status == 429:
                            wait = base_delay * (2 ** attempt)
                            time.sleep(wait)
                            continue
                        if attempt < 2:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        raise
                    except Exception:
                        # cleanup partial
                        try:
                            if temp_path and temp_path.exists():
                                temp_path.unlink()
                        except Exception:
                            pass
                        if attempt < 2:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        raise
                # if somehow not returned, raise
                raise RuntimeError("video download failed after retries")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(download_video, url): url for url in video_urls}
                for future in futures:
                    if should_stop():
                        log_progress("중지 요청을 감지했습니다. 비디오 다운로드를 중단합니다.")
                        break
                    try:
                        url, local_name, is_image_fallback = future.result()
                        video_to_local[url] = f"videos/{local_name}"
                        video_fallback_image[url] = is_image_fallback
                        videos_downloaded += 1
                    except Exception as exc:  # pragma: no cover - network-dependent
                        err_msg = f"[{aid}] 비디오 다운로드 실패: {futures[future]} ({exc})"
                        errors.append(err_msg)
                        log_progress(err_msg, level=logging.WARNING)
                    # per-download throttle to avoid bursts
                    time.sleep(0.3)
            vid_ms = (time.perf_counter() - vid_start) * 1000

        if video_poster_map and not should_stop():
            poster_start = time.perf_counter()
            posters_dir = post_dir / "videos"

            def download_poster(target_url: str):
                headers = {
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Referer": post_url,
                }
                response = session.get(target_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
                content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()

                parsed = urlparse(target_url)
                name = Path(parsed.path).name or "poster"
                safe_name = safe_filename(name) or "poster"
                ext = None
                if content_type:
                    guessed_ext = mimetypes.guess_extension(content_type)
                    if guessed_ext:
                        ext = guessed_ext
                if not ext:
                    ext = Path(safe_name).suffix or ".jpg"
                if not safe_name.lower().endswith(ext.lower()):
                    safe_name = f"{Path(safe_name).stem}{ext}"

                dest = posters_dir / safe_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)

                return target_url, dest.name

            with ThreadPoolExecutor(max_workers=min(3, len(video_poster_map))) as pool:
                futures = {pool.submit(download_poster, poster_url): video_src for video_src, poster_url in video_poster_map.items()}
                for future in futures:
                    if should_stop():
                        log_progress("중지 요청을 감지했습니다. 포스터 다운로드를 중단합니다.")
                        break
                    try:
                        poster_url, local_name = future.result()
                        video_src_for_poster = futures[future]
                        poster_to_local[video_src_for_poster] = f"videos/{local_name}"
                    except Exception as exc:  # pragma: no cover - network-dependent
                        err_msg = f"[{aid}] 비디오 포스터 다운로드 실패: {futures[future]} ({exc})"
                        errors.append(err_msg)
                        log_progress(err_msg, level=logging.WARNING)
                    # per-download throttle to avoid bursts
                    time.sleep(0.3)
            poster_ms = (time.perf_counter() - poster_start) * 1000

        if should_stop():
            log_progress("중지 요청을 감지했습니다. 게시물 저장을 중단합니다.")
            break

        rewritten = ""
        rewrite_start = time.perf_counter()
        if content_html:
            soup = BeautifulSoup(content_html, "lxml")
            
            # 외부 링크 unsafelink.com 제거
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                if href.startswith("https://unsafelink.com/"):
                    anchor["href"] = href.replace("https://unsafelink.com/", "", 1)

            for img in soup.find_all("img"):
                raw_url = img.get("src") or img.get("data-src") or img.get("data-original")
                if not raw_url:
                    continue
                full = urljoin(post_url, raw_url)
                if full in url_to_local:
                    img["src"] = url_to_local[full]
                    for key in ("data-src", "data-original"):
                        if key in img.attrs:
                            del img.attrs[key]

            for video in soup.find_all("video"):
                v_src = (
                    video.get("data-originalurl")
                    or video.get("src")
                    or video.get("data-src")
                    or video.get("data-original")
                )
                if v_src:
                    full = urljoin(post_url, v_src)
                    if full in video_to_local:
                        local_src = video_to_local[full]
                        if video_fallback_image.get(full):
                            img = soup.new_tag("img", src=local_src)
                            # preserve width/height/style/class if present
                            for key in ("width", "height", "style", "class"):
                                if key in video.attrs:
                                    img.attrs[key] = video.attrs[key]
                            video.replace_with(img)
                            continue
                        video["src"] = local_src
                        poster_local = poster_to_local.get(full)
                        if poster_local:
                            video["poster"] = poster_local
                for key in ("data-originalurl", "data-src", "data-original"):
                    if key in video.attrs:
                        del video.attrs[key]
                for source in video.find_all("source"):
                    s = source.get("src") or source.get("data-src") or source.get("data-original")
                    if s:
                        full = urljoin(post_url, s)
                        if full in video_to_local:
                            local_src = video_to_local[full]
                            if video_fallback_image.get(full):
                                source.decompose()
                                continue
                            source["src"] = local_src
                    for key in ("data-src", "data-original"):
                        if key in source.attrs:
                            del source.attrs[key]
            rewritten = str(soup)

        offline_html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans KR", sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }}
    img {{ max-width: 100%; height: auto; }}
    .meta {{ color: #666; font-size: 14px; margin-bottom: 16px; }}
    .meta a {{ color: inherit; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">원문: <a href="{post_url}">{post_url}</a></div>
  <div class="meta">작성자: {author or '-'}</div>
  <article>{rewritten}</article>
</body>
</html>
"""
        with open(post_dir / "index.html", "w", encoding="utf-8") as handle:
            handle.write(offline_html)

        rewrite_ms = (time.perf_counter() - rewrite_start) * 1000

        manifest_map[aid] = {
            "id": aid,
            "title": title,
            "url": post_url,
            "dir": post_dir.name,
            "images": len(image_urls),
            "downloaded_images": len(url_to_local),
            "videos": len(video_urls),
            "downloaded_videos": len(video_to_local),
            "video_fallback_images": sum(1 for v in video_urls if video_fallback_image.get(v)),
        }
        total_ms = (time.perf_counter() - post_start) * 1000
        log_progress(
            f"[{aid}] timings fetch={fetch_ms:.0f}ms img={img_ms:.0f}ms vid={vid_ms:.0f}ms poster={poster_ms:.0f}ms save={rewrite_ms:.0f}ms total={total_ms:.0f}ms"
        )
        posts_saved += 1
        if should_stop():
            log_progress("중지 요청을 감지했습니다. 이후 게시물 처리를 중단합니다.")
            break
        time.sleep(config.sleep)

    manifest = list(manifest_map.values())
    with open(config.out_dir / "_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    finished_at = time.time()
    run_meta = {
        "channel": config.channel,
        "category": config.category,
        "start_page": config.start_page,
        "end_page": config.end_page,
        "resolved_end_page": resolved_end_page,
        "auto_last_page": auto_last_page,
        "sleep": config.sleep,
        "started_at": started_at,
        "finished_at": finished_at,
        "posts_saved": posts_saved,
        "images_downloaded": images_downloaded,
        "videos_downloaded": videos_downloaded,
        "pages_processed": pages_processed,
        "errors": len(errors),
    }
    with open(config.out_dir / "_run.json", "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, ensure_ascii=False, indent=2)

    if should_stop():
        log_progress(f"백업 중단됨: 현재까지 {posts_saved}개 저장, {images_downloaded}개 이미지 다운로드.")
    else:
        log_progress(f"백업 완료: 총 {posts_saved}개 저장됨.")

    return BackupResult(
        manifest=manifest,
        errors=errors,
        out_dir=config.out_dir,
        pages_processed=pages_processed,
        posts_saved=posts_saved,
        images_downloaded=images_downloaded,
        videos_downloaded=videos_downloaded,
        duration=finished_at - started_at,
        started_at=started_at,
        finished_at=finished_at,
        end_page=resolved_end_page,
        auto_last_page=auto_last_page,
    )
