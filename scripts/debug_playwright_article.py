import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z._-]+", "_", value or "").strip("_")
    return value[:80] or "debug"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _try_get_text(page, selector: str) -> Optional[str]:
    try:
        loc = page.locator(selector)
        if loc.count() <= 0:
            return None
        return (loc.first.inner_text() or "").strip()
    except Exception:
        return None


def _try_get_html(page, selector: str) -> Optional[str]:
    try:
        loc = page.locator(selector)
        if loc.count() <= 0:
            return None
        return loc.first.inner_html()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Playwright로 아카라이브 게시물 페이지를 열고, 실제 렌더링 결과/네트워크를 디버깅 저장합니다.",
    )
    parser.add_argument("--channel", default="3d3d")
    parser.add_argument("--id", type=int, default=6457546)
    parser.add_argument("--out", default="playwright_debug")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--storage-state", default="", help="로그인/쿠키가 담긴 storage_state JSON 경로(선택)")
    parser.add_argument("--save-storage-state", action="store_true", help="실행 후 storage_state를 out에 저장")
    args = parser.parse_args()

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(
            "Playwright가 설치되어 있지 않습니다. "
            "`pip install playwright` 후 `python -m playwright install chromium`을 실행하세요.",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 2

    channel = args.channel.strip()
    aid = args.id
    url = f"https://arca.live/b/{channel}/{aid}"
    out_root = Path(args.out) / f"{_safe_slug(channel)}_{aid}"
    out_root.mkdir(parents=True, exist_ok=True)

    requests_log: List[Dict[str, Any]] = []
    responses_log: List[Dict[str, Any]] = []
    failed_log: List[Dict[str, Any]] = []
    console_log: List[Dict[str, Any]] = []
    page_errors: List[str] = []

    started_at = time.time()
    result: Dict[str, Any] = {
        "url": url,
        "final_url": None,
        "status": None,
        "title": None,
        "detected": {},
        "timing": {},
    }

    def on_request(req):  # pragma: no cover - runtime
        try:
            requests_log.append(
                {
                    "url": req.url,
                    "method": req.method,
                    "resource_type": req.resource_type,
                }
            )
        except Exception:
            return

    def on_response(res):  # pragma: no cover - runtime
        try:
            responses_log.append(
                {
                    "url": res.url,
                    "status": res.status,
                    "resource_type": res.request.resource_type,
                    "content_type": (res.headers or {}).get("content-type"),
                }
            )
        except Exception:
            return

    def on_request_failed(req):  # pragma: no cover - runtime
        try:
            failed_log.append(
                {
                    "url": req.url,
                    "method": req.method,
                    "resource_type": req.resource_type,
                    "failure": (req.failure or {}).get("errorText"),
                }
            )
        except Exception:
            return

    def on_console(msg):  # pragma: no cover - runtime
        try:
            console_log.append({"type": msg.type, "text": msg.text})
        except Exception:
            return

    def on_page_error(exc):  # pragma: no cover - runtime
        try:
            page_errors.append(str(exc))
        except Exception:
            return

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context_kwargs: Dict[str, Any] = {
            "user_agent": DEFAULT_USER_AGENT,
            "locale": "ko-KR",
            "extra_http_headers": {"Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"},
        }
        if args.storage_state:
            context_kwargs["storage_state"] = args.storage_state
        context = browser.new_context(**context_kwargs)

        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)
        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)
        page.on("console", on_console)
        page.on("pageerror", on_page_error)

        main_response = None
        try:
            main_response = page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        except PlaywrightTimeoutError as exc:
            result["detected"]["goto_timeout"] = str(exc)
        except PlaywrightError as exc:
            result["detected"]["goto_error"] = str(exc)

        status = None
        if main_response is not None:
            try:
                status = main_response.status
            except Exception:
                status = None
        result["status"] = status

        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass

        try:
            result["final_url"] = page.url
        except Exception:
            result["final_url"] = None

        try:
            result["title"] = page.title()
        except Exception:
            result["title"] = None

        # 저장 (항상 남기기)
        html_path = out_root / "page.html"
        shot_path = out_root / "page.png"
        network_path = out_root / "network.json"
        summary_path = out_root / "summary.json"

        try:
            html = page.content()
            html_path.write_text(html, encoding="utf-8")
        except Exception as exc:
            result["detected"]["html_save_error"] = str(exc)

        try:
            page.screenshot(path=str(shot_path), full_page=True)
        except Exception as exc:
            result["detected"]["screenshot_error"] = str(exc)

        _write_json(
            network_path,
            {
                "url": url,
                "final_url": result["final_url"],
                "status": status,
                "requests": requests_log,
                "responses": responses_log,
                "request_failed": failed_log,
                "console": console_log,
                "page_errors": page_errors,
            },
        )

        # 페이지에서 실제로 뭘 봤는지 요약
        body_text = _try_get_text(page, "body") or ""
        body_text_compact = re.sub(r"\s+", " ", body_text).strip()
        result["detected"]["body_snippet"] = body_text_compact[:400]
        result["detected"]["has_permission_text"] = (
            ("권한" in body_text) or ("permission" in body_text.lower()) or ("접근" in body_text)
        )
        result["detected"]["has_cloudflare"] = ("Just a moment" in body_text) or ("cf-" in (html or ""))

        selectors = [
            "article",
            "article.board-article",
            "div.board-article",
            "div.fr-view",
            "div.article-content",
            "main",
        ]
        extracted: Dict[str, Any] = {"matched_selector": None}
        for sel in selectors:
            html_candidate = _try_get_html(page, sel)
            if html_candidate:
                extracted["matched_selector"] = sel
                extracted["content_html_len"] = len(html_candidate)
                extracted["content_html_snippet"] = html_candidate[:500]
                break
        result["extracted"] = extracted

        result["timing"]["seconds"] = round(time.time() - started_at, 3)
        result["artifacts"] = {
            "html": str(html_path),
            "screenshot": str(shot_path),
            "network": str(network_path),
            "summary": str(summary_path),
        }

        if args.save_storage_state:
            try:
                state = context.storage_state()
                _write_json(out_root / "storage_state.json", state)
                result["artifacts"]["storage_state"] = str(out_root / "storage_state.json")
            except Exception as exc:
                result["detected"]["storage_state_error"] = str(exc)

        _write_json(summary_path, result)
        try:
            context.close()
            browser.close()
        except Exception:
            pass

    print(f"[DONE] {url}")
    print(f"- status: {result.get('status')}")
    print(f"- title: {result.get('title')}")
    print(f"- final_url: {result.get('final_url')}")
    print(f"- body_snippet: {result.get('detected', {}).get('body_snippet')}")
    print(f"- artifacts: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

