import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from urllib.parse import parse_qs, urlencode

from flask import Blueprint, Response, current_app, render_template, request, stream_with_context

from .services.arca_backup import BackupConfig, backup_channel, safe_filename

main = Blueprint('main', __name__)


# 단일 작업 상태 관리
_job_state = {
    "running": False,
    "thread": None,
    "history": [],            # [{"kind": log|summary|error|refresh|done, "payload": str}]
    "result": None,           # BackupResult
    "params": None,
    "subscribers": set(),     # SSE 구독 큐
    "stop_event": None,       # threading.Event
    "lock": threading.Lock(),
}


def _broadcast(event: dict) -> None:
    """이벤트를 기록하고 모든 구독자 큐에 전달한다."""
    with _job_state["lock"]:
        _job_state["history"].append(event)
        if len(_job_state["history"]) > 500:
            _job_state["history"] = _job_state["history"][-500:]
        for q in list(_job_state["subscribers"]):
            q.put(event)


def _format_log_html(msg: str) -> str:
    return f"<div class='py-1 border-b border-gray-700 text-sm'>{msg}</div>"


def _format_summary_html(res) -> str:
    if res is None:
        return ""
    return (
        "<div class='mt-4 p-3 bg-slate-800 rounded border border-emerald-900/50'>"
        "<div class='font-bold text-emerald-400 mb-1'>[백업 완료 리포트]</div>"
        "<div class='grid grid-cols-2 gap-x-4 gap-y-1 text-xs'>"
        f"<span class='text-slate-400'>저장된 게시글:</span> <span class='text-slate-200'>{res.posts_saved}개</span>"
        f"<span class='text-slate-400'>다운로드 이미지:</span> <span class='text-slate-200'>{res.images_downloaded}개</span>"
        f"<span class='text-slate-400'>다운로드 비디오:</span> <span class='text-slate-200'>{getattr(res, 'videos_downloaded', 0)}개</span>"
        f"<span class='text-slate-400'>소요 시간:</span> <span class='text-slate-200'>{res.duration:.1f}초</span>"
        f"<span class='text-slate-400'>발생 에러:</span> <span class='text-{'red-400' if res.errors else 'slate-200'}'>{len(res.errors)}건</span>"
        "</div>"
        "</div>"
    )


def _parse_int(value, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, minimum)


def _parse_float(value, default: float, minimum: float = 0.0, maximum: float = 5.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _build_out_dir(base_root: Path, out_name: str, channel: str, category: str) -> Path:
    safe_channel = safe_filename(channel)
    safe_category = safe_filename(category)
    slug = safe_filename(out_name) or f"arca_backup_{safe_channel}{f'_{safe_category}' if safe_category else ''}"
    if not slug:
        slug = "arca_backup"
    return base_root / slug


def _load_runs(base_root: Path):
    runs = []
    for manifest_path in base_root.glob("*/_manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            continue

        run_meta = {}
        run_meta_path = manifest_path.parent / "_run.json"
        if run_meta_path.exists():
            try:
                with open(run_meta_path, "r", encoding="utf-8") as handle:
                    run_meta = json.load(handle) or {}
            except Exception:
                run_meta = {}

        items = []
        if isinstance(manifest, list):
            items = manifest
        elif isinstance(manifest, dict):
            candidate = manifest.get("items") or manifest.get("manifest") or []
            if isinstance(candidate, list):
                items = candidate

        updated_at = datetime.fromtimestamp(manifest_path.stat().st_mtime)
        runs.append(
            {
                "name": manifest_path.parent.name,
                "path": str(manifest_path.parent),
                "updated_at": updated_at,
                "channel": (run_meta.get("channel") or "").strip() or None,
                "category": (run_meta.get("category") or "").strip() or None,
                "posts": len(items),
                "images": sum(item.get("downloaded_images", 0) for item in items if isinstance(item, dict)),
            }
        )

    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return runs


def _extract_filters(runs):
    channels = sorted({run.get("channel") for run in runs if run.get("channel")})
    categories = sorted({run.get("category") for run in runs if run.get("category")})
    return channels, categories


def _load_runs(base_root: Path):
    runs = []
    for manifest_path in base_root.glob("*/_manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            continue

        run_meta = {}
        run_meta_path = manifest_path.parent / "_run.json"
        if run_meta_path.exists():
            try:
                with open(run_meta_path, "r", encoding="utf-8") as handle:
                    run_meta = json.load(handle) or {}
            except Exception:
                run_meta = {}

        items = []
        if isinstance(manifest, list):
            items = manifest
        elif isinstance(manifest, dict):
            candidate = manifest.get("items") or manifest.get("manifest") or []
            if isinstance(candidate, list):
                items = candidate

        updated_at = datetime.fromtimestamp(manifest_path.stat().st_mtime)
        runs.append(
            {
                "name": manifest_path.parent.name,
                "path": str(manifest_path.parent),
                "updated_at": updated_at,
                "channel": (run_meta.get("channel") or "").strip() or None,
                "category": (run_meta.get("category") or "").strip() or None,
                "posts": len(items),
                "images": sum(item.get("downloaded_images", 0) for item in items if isinstance(item, dict)),
            }
        )

    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return runs


def _extract_filters(runs):
    channels = sorted({run.get("channel") for run in runs if run.get("channel")})
    categories = sorted({run.get("category") for run in runs if run.get("category")})
    return channels, categories


@main.route('/')
def index():
    defaults = {
        "channel": "3d3d",
        "category": "",
        "start_page": 1,
        "end_page": 3,
        "sleep": 0.8,
        "out_name": "arca_backup_3d3d",
    }

    with _job_state["lock"]:
        active_job = _job_state["running"]
        active_params_qs = _job_state.get("params")

    active_params = dict(defaults)
    if active_params_qs:
        parsed = parse_qs(active_params_qs)

        def pick(key, fallback):
            values = parsed.get(key) or []
            return values[0] if values else fallback

        active_params.update(
            {
                "channel": pick("channel", active_params["channel"]),
                "category": pick("category", active_params.get("category", "")),
                "start_page": pick("start_page", active_params["start_page"]),
                "end_page": pick("end_page", active_params["end_page"]),
                "sleep": pick("sleep", active_params["sleep"]),
                "out_name": pick("out_name", active_params.get("out_name", "")),
            }
        )

    backup_root: Path = current_app.config['BACKUP_ROOT']
    runs = _load_runs(backup_root)
    channels, categories = _extract_filters(runs)
    return render_template(
        'index.html',
        defaults=defaults,
        active_job=active_job,
        active_params=active_params,
        active_params_qs=active_params_qs,
        runs=runs,
        backup_root=str(backup_root),
        channels=channels,
        categories=categories,
    )


@main.route('/backups/list')
def backups_list():
    backup_root: Path = current_app.config['BACKUP_ROOT']
    runs = _load_runs(backup_root)

    channel = (request.args.get("channel") or "").strip()
    category = (request.args.get("category") or "").strip()

    if channel:
        runs = [run for run in runs if run.get("channel") == channel]
    if category:
        runs = [run for run in runs if run.get("category") == category]

    return render_template('partials/run_list.html', runs=runs)


@main.route('/backup/run', methods=['POST'])
def run_backup():
    form = request.form
    channel = (form.get("channel") or "").strip() or "3d3d"
    category = (form.get("category") or "").strip()
    start_page = _parse_int(form.get("start_page"), 1, 1)
    raw_end_page = form.get("end_page")
    end_page = None
    if raw_end_page is not None and raw_end_page.strip() != "":
        end_page = _parse_int(raw_end_page, start_page, 1)
    sleep = _parse_float(form.get("sleep"), 0.8, minimum=0.0, maximum=5.0)
    out_name = (form.get("out_name") or "").strip()

    params = urlencode({
        "channel": channel,
        "category": category,
        "start_page": start_page,
        "end_page": "" if end_page is None else end_page,
        "sleep": sleep,
        "out_name": out_name
    })

    # 백업 실행 요청을 즉시 처리 (이미 실행 중이면 건너뜀)
    def start_job():
        with _job_state["lock"]:
            if _job_state["running"]:
                return False
            _job_state["running"] = True
            _job_state["history"] = []
            _job_state["result"] = None
            _job_state["params"] = params
            _job_state["stop_event"] = threading.Event()

        backup_root: Path = current_app.config['BACKUP_ROOT']
        out_dir = _build_out_dir(backup_root, out_name, channel, category)

        config = BackupConfig(
            channel=channel,
            category=category,
            start_page=start_page,
            end_page=end_page,
            out_dir=out_dir,
            sleep=sleep,
        )

        def worker():
            try:
                _broadcast({"kind": "log", "payload": _format_log_html("[시스템] 백업 프로세스를 시작합니다...")})

                def cb(msg):
                    _broadcast({"kind": "log", "payload": _format_log_html(msg)})

                res = backup_channel(config, progress_callback=cb, stop_event=_job_state["stop_event"])
                with _job_state["lock"]:
                    _job_state["result"] = res

                _broadcast({"kind": "summary", "payload": _format_summary_html(res)})
                _broadcast({"kind": "log", "payload": _format_log_html("[시스템] 모든 작업이 완료되었습니다.")})
            except Exception as exc:  # pragma: no cover - runtime safety
                _broadcast({"kind": "error", "payload": _format_log_html(f"[오류] {exc}")})
            finally:
                _broadcast({"kind": "refresh", "payload": "<script>htmx.trigger('#run-history', 'refresh-runs')</script>"})
                _broadcast({"kind": "done"})
                with _job_state["lock"]:
                    _job_state["running"] = False
                    _job_state["thread"] = None
                    _job_state["stop_event"] = None

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        with _job_state["lock"]:
            _job_state["thread"] = t
        return True

    start_job()

    return render_template('partials/sse_container.html', params=params)


@main.route('/backup/stop', methods=['POST'])
def stop_backup():
    with _job_state["lock"]:
        evt = _job_state.get("stop_event")
        running = _job_state["running"]

    if running and evt is not None:
        evt.set()
        _broadcast({"kind": "log", "payload": _format_log_html("[시스템] 중지 요청을 보냈습니다.")})
        return {"status": "stopping"}

    return {"status": "idle"}


@main.route('/backup/run_sse')
def run_backup_sse():
    @stream_with_context
    def generate():
        q: SimpleQueue = SimpleQueue()

        def serialize(evt: dict):
            payload = evt.get("payload")
            if not payload:
                return None
            return f"data: {str(payload).replace(chr(10), '')}\n\n"

        with _job_state["lock"]:
            history = list(_job_state["history"])
            _job_state["subscribers"].add(q)

        try:
            for evt in history:
                chunk = serialize(evt)
                if chunk:
                    yield chunk

            while True:
                evt = q.get()
                chunk = serialize(evt)
                if chunk:
                    yield chunk
        finally:
            with _job_state["lock"]:
                _job_state["subscribers"].discard(q)

    return Response(generate(), mimetype='text/event-stream')
