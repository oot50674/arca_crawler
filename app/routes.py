import json
from datetime import datetime
from pathlib import Path
import threading
from queue import SimpleQueue

from flask import Blueprint, current_app, render_template, request, Response, stream_with_context
import logging

from .services.arca_backup import BackupConfig, backup_channel, safe_filename

main = Blueprint('main', __name__)


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
    backup_root: Path = current_app.config['BACKUP_ROOT']
    runs = _load_runs(backup_root)
    channels, categories = _extract_filters(runs)
    return render_template(
        'index.html',
        defaults=defaults,
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
    end_page = _parse_int(form.get("end_page"), start_page, 1)
    sleep = _parse_float(form.get("sleep"), 0.8, minimum=0.0, maximum=5.0)
    out_name = (form.get("out_name") or "").strip()

    from urllib.parse import urlencode
    params = urlencode({
        "channel": channel,
        "category": category,
        "start_page": start_page,
        "end_page": end_page,
        "sleep": sleep,
        "out_name": out_name
    })

    return render_template('partials/sse_container.html', params=params)


@main.route('/backup/run_sse')
def run_backup_sse():
    channel = (request.args.get("channel") or "").strip() or "3d3d"
    category = (request.args.get("category") or "").strip()
    start_page = _parse_int(request.args.get("start_page"), 1, 1)
    end_page = _parse_int(request.args.get("end_page"), start_page, 1)
    sleep = _parse_float(request.args.get("sleep"), 0.8, minimum=0.0, maximum=5.0)
    out_name = (request.args.get("out_name") or "").strip()

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

    @stream_with_context
    def generate():
        q: SimpleQueue = SimpleQueue()
        done = object()
        result_holder = {}

        def callback(msg: str):
            q.put({"type": "log", "msg": msg})

        def worker():
            try:
                res = backup_channel(config, progress_callback=callback)
                result_holder["res"] = res
                q.put({"type": "summary"})
            except Exception as exc:  # pragma: no cover - runtime safety
                q.put({"type": "error", "msg": str(exc)})
            finally:
                q.put(done)

        threading.Thread(target=worker, daemon=True).start()

        # initial notice
        yield "data: <div class='font-bold text-blue-400'>[시스템] 백업 프로세스를 시작합니다...</div>\n\n"

        while True:
            item = q.get()
            if item is done:
                break

            if item.get("type") == "log":
                yield f"data: <div class='py-1 border-b border-gray-700 text-sm'>{item['msg']}</div>\n\n"
            elif item.get("type") == "error":
                yield f"data: <div class='font-bold text-red-400'>[오류] {item.get('msg','')}</div>\n\n"
            elif item.get("type") == "summary":
                res = result_holder.get("res")
                if res:
                    summary = f"""
<div class='mt-4 p-3 bg-slate-800 rounded border border-emerald-900/50'>
    <div class='font-bold text-emerald-400 mb-1'>[백업 완료 리포트]</div>
    <div class='grid grid-cols-2 gap-x-4 gap-y-1 text-xs'>
        <span class='text-slate-400'>저장된 게시글:</span> <span class='text-slate-200'>{res.posts_saved}개</span>
        <span class='text-slate-400'>다운로드 이미지:</span> <span class='text-slate-200'>{res.images_downloaded}개</span>
        <span class='text-slate-400'>소요 시간:</span> <span class='text-slate-200'>{res.duration:.1f}초</span>
        <span class='text-slate-400'>발생 에러:</span> <span class='text-{"red-400" if res.errors else "slate-200"}'>{len(res.errors)}건</span>
    </div>
</div>
"""
                    yield f"data: {summary.replace(chr(10), '')}\n\n"
                yield "data: <div class='font-bold text-green-400 mt-2'>[시스템] 모든 작업이 완료되었습니다.</div>\n\n"
                yield "data: <script>htmx.trigger('#run-history', 'refresh-runs')</script>\n\n"

    return Response(generate(), mimetype='text/event-stream')
