import json
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, render_template, request
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
    if end_page < start_page:
        end_page = start_page
    sleep = _parse_float(form.get("sleep"), 0.8, minimum=0.0, maximum=5.0)
    out_name = (form.get("out_name") or "").strip()

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

    current_app.logger.info(
        "[backup] start channel=%s category=%s pages=%s-%s sleep=%.2f out=%s",
        channel,
        category or "(all)",
        start_page,
        end_page,
        sleep,
        out_dir,
    )
    result = backup_channel(config)
    runs = _load_runs(backup_root)
    channels, categories = _extract_filters(runs)
    current_app.logger.info(
        "[backup] done channel=%s category=%s posts=%s images=%s errors=%s",
        channel,
        category or "(all)",
        result.posts_saved,
        result.images_downloaded,
        len(result.errors),
    )
    return render_template(
        'partials/backup_result.html',
        result=result,
        runs=runs,
        backup_root=str(backup_root),
        channels=channels,
        categories=categories,
    )
