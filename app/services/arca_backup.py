import base64
import json
import logging
import mimetypes
import os
import re
import time
import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
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
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        urls.append(urljoin(base_url, src))

    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        path = urlparse(full).path.lower()
        if any(path.endswith(ext) for ext in IMAGE_EXTS):
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


def build_channel_url(channel: str, category: str, page: int) -> str:
    base = urljoin(BASE_URL, f"b/{channel}")
    params = {"p": page}
    if category:
        params["category"] = category
    return f"{base}?{urlencode(params)}"


def build_article_url(channel: str, aid: int) -> str:
    return urljoin(BASE_URL, f"b/{channel}/{aid}")


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
    ) -> List[int]:
        self._ensure_started()

        if self._page is None:
            return []

        url = build_channel_url(channel=channel, category=category, page=page)
        request_events: List[Dict[str, Any]] = []
        response_events: List[Dict[str, Any]] = []
        error_events: List[Dict[str, Any]] = []
        json_ids: Set[int] = set()

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
                if ids:
                    return ids
                if json_ids:
                    return list(sorted(json_ids))
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
    end_page: int
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
    duration: float
    started_at: float
    finished_at: float


def backup_channel(config: BackupConfig, progress_callback=None) -> BackupResult:
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
    manifest: List[Dict] = []
    errors: List[str] = []
    posts_saved = 0
    images_downloaded = 0
    pages_processed = 0
    page_listings: List[Dict[str, Any]] = []
    all_ids: List[int] = []

    config.out_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"백업 시작: {config.channel} (페이지 {config.start_page}~{config.end_page})")

    # 1) Listing 수집 단계
    for page in range(config.start_page, config.end_page + 1):
        pages_processed += 1
        log_progress(f"목록 수집 중: {page}페이지...")
        playwright_error: Optional[Exception] = None
        ids: List[int] = []

        def collect_ids(target_category: str) -> List[int]:
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
            ids = collect_ids(config.category)
        except Exception as exc:  # pragma: no cover - network-dependent
            playwright_error = exc
            logger.warning("[page %s] playwright listing 실패: %s", page, playwright_error)

        listing_error: Optional[Exception] = None
        detail = ""
        if not ids:
            listing_error = RuntimeError("목록에서 글 ID를 발견할 수 없습니다.")

        if not ids:
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
        time.sleep(config.sleep)

    # listing 결과 저장
    with open(config.out_dir / "_listing.json", "w", encoding="utf-8") as handle:
        json.dump(page_listings, handle, ensure_ascii=False, indent=2)

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
                
                existing_posts[aid_val] = {
                    "id": aid_val,
                    "title": post_data.get("title", p.name),
                    "url": post_data.get("url", ""),
                    "dir": p.name,
                    "images": 0, # 정확한 숫자는 알 수 없으나 스킵에는 지장 없음
                    "downloaded_images": 0,
                }
            except Exception:
                continue

    for idx, aid in enumerate(all_ids, 1):
        if aid in existing_posts:
            item = existing_posts[aid]
            # 실제 폴더와 최종 결과물(index.html)이 존재하는지 한 번 더 확인
            target_dir = config.out_dir / item.get("dir", "")
            if (target_dir / "index.html").exists():
                log_progress(f"[{idx}/{total_ids}] [{aid}] 이미 완료된 게시물입니다. 스킵합니다.")
                manifest.append(item)
                continue
            else:
                log_progress(f"[{idx}/{total_ids}] [{aid}] 매니페스트에는 있으나 결과물 파일이 없습니다. 다시 크롤링합니다.")

        log_progress(f"[{idx}/{total_ids}] [{aid}] 크롤링 중...")
        try:
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
        url_to_local: Dict[str, str] = {}

        for url in image_urls:
            try:
                data_url = download_as_data_url(url, referer=post_url, session=session)
                url_to_local[url] = data_url
                images_downloaded += 1
            except Exception as exc:  # pragma: no cover - network-dependent
                errors.append(f"[{aid}] 이미지 다운로드 실패: {url} ({exc})")

        rewritten = ""
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

        manifest.append(
            {
                "id": aid,
                "title": title,
                "url": post_url,
                "dir": post_dir.name,
                "images": len(image_urls),
                "downloaded_images": len(url_to_local),
            }
        )
        posts_saved += 1
        time.sleep(config.sleep)

    with open(config.out_dir / "_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    finished_at = time.time()
    run_meta = {
        "channel": config.channel,
        "category": config.category,
        "start_page": config.start_page,
        "end_page": config.end_page,
        "sleep": config.sleep,
        "started_at": started_at,
        "finished_at": finished_at,
        "posts_saved": posts_saved,
        "images_downloaded": images_downloaded,
        "pages_processed": pages_processed,
        "errors": len(errors),
    }
    with open(config.out_dir / "_run.json", "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, ensure_ascii=False, indent=2)

    log_progress(f"백업 완료: 총 {posts_saved}개 저장됨.")

    return BackupResult(
        manifest=manifest,
        errors=errors,
        out_dir=config.out_dir,
        pages_processed=pages_processed,
        posts_saved=posts_saved,
        images_downloaded=images_downloaded,
        duration=finished_at - started_at,
        started_at=started_at,
        finished_at=finished_at,
    )
