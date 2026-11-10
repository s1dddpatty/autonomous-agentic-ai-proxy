import os
import shutil
import sqlite3
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from typing import List

import requests

from src.core.config import DATA_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.core.database import get_config, set_config

MAX_TELEGRAM_ZIP_BYTES = 40 * 1024 * 1024  # 40 MB (safely below Telegram's 50 MB Bot API limit)
DEFAULT_BACKUP_INTERVAL_DAYS = int(os.getenv("BACKUP_INTERVAL_DAYS", "30"))


def _safe_sqlite_copy(src_path: str, dst_path: str) -> None:
    """Safely copies a live SQLite database (including WAL changes) using SQLite's backup API."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    src_conn = sqlite3.connect(src_path, timeout=30.0)
    dst_conn = sqlite3.connect(dst_path)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _stage_data_snapshot(staging_dir: str) -> List[str]:
    """Stages a consistent snapshot of DATA_DIR files and returns relative file paths."""
    if not os.path.exists(DATA_DIR):
        return []

    sqlite_names = {"chats.db", "checkpoint.sqlite", "chroma.sqlite3"}
    staged_files: List[str] = []

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            # Skip SQLite journal/WAL/SHM files because backup() merges them cleanly
            if fname.endswith(("-wal", "-shm", "-journal")):
                continue

            src_file = os.path.join(root, fname)
            rel_path = os.path.relpath(src_file, DATA_DIR)
            dst_file = os.path.join(staging_dir, rel_path)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            if fname in sqlite_names:
                _safe_sqlite_copy(src_file, dst_file)
            else:
                shutil.copy2(src_file, dst_file)

            staged_files.append(rel_path)

    return sorted(staged_files)


def _zip_file_groups(staging_dir: str, staged_files: List[str], out_dir: str, label: str, tag: str) -> List[str]:
    """Packages staged files into one or more ZIPs, each kept under MAX_TELEGRAM_ZIP_BYTES."""
    if not staged_files:
        return []

    priority_files = [f for f in staged_files if not f.startswith("chroma_db")]
    chroma_files = [f for f in staged_files if f.startswith("chroma_db")]

    zip_paths: List[str] = []

    def _pack_group(files_subset: List[str], group_name: str) -> None:
        if not files_subset:
            return
        part_idx = 1
        current_path = os.path.join(out_dir, f"proxy_{group_name}_{tag}_part{part_idx}.zip")
        zf = zipfile.ZipFile(current_path, "w", compression=zipfile.ZIP_DEFLATED)
        group_zips: List[str] = []
        try:
            for i, rel_path in enumerate(files_subset):
                full_path = os.path.join(staging_dir, rel_path)
                zf.write(full_path, arcname=os.path.join("data", rel_path))
                if os.path.getsize(current_path) >= MAX_TELEGRAM_ZIP_BYTES and i < len(files_subset) - 1:
                    zf.close()
                    group_zips.append(current_path)
                    part_idx += 1
                    current_path = os.path.join(out_dir, f"proxy_{group_name}_{tag}_part{part_idx}.zip")
                    zf = zipfile.ZipFile(current_path, "w", compression=zipfile.ZIP_DEFLATED)
        finally:
            zf.close()

        group_zips.append(current_path)
        if len(group_zips) == 1:
            clean_path = os.path.join(out_dir, f"proxy_{group_name}_{tag}.zip")
            os.replace(group_zips[0], clean_path)
            zip_paths.append(clean_path)
        else:
            zip_paths.extend(group_zips)

    _pack_group(priority_files, "core")
    _pack_group(chroma_files, "chroma")
    return zip_paths


def _split_oversized_file(file_path: str, out_dir: str) -> List[str]:
    """Splits a single archive file into 40 MB chunks if a single raw file caused it to exceed the cap."""
    if os.path.getsize(file_path) <= MAX_TELEGRAM_ZIP_BYTES:
        return [file_path]

    chunk_paths: List[str] = []
    base = os.path.basename(file_path)
    with open(file_path, "rb") as src:
        idx = 1
        while True:
            chunk = src.read(MAX_TELEGRAM_ZIP_BYTES)
            if not chunk:
                break
            chunk_path = os.path.join(out_dir, f"{base}.{idx:03d}")
            with open(chunk_path, "wb") as dst:
                dst.write(chunk)
            chunk_paths.append(chunk_path)
            idx += 1

    os.remove(file_path)
    return chunk_paths


def send_telegram_backup(reason: str = "Scheduled Monthly Backup") -> tuple[bool, str]:
    """
    Snapshots DATA_DIR safely, packages it into Telegram-safe archive(s),
    and delivers them to AUTHORIZED_TG_USER_ID / TELEGRAM_CHAT_ID.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID."

    tag = datetime.now().strftime("%Y-%m-%d_%H%M")
    tmp_root = tempfile.mkdtemp(prefix="proxy_backup_")
    staging_dir = os.path.join(tmp_root, "staging")
    out_dir = os.path.join(tmp_root, "archives")
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        staged = _stage_data_snapshot(staging_dir)
        if not staged:
            return False, "No files found in data directory to back up."

        raw_zips = _zip_file_groups(staging_dir, staged, out_dir, "data_backup", tag)
        upload_files: List[str] = []
        for zp in raw_zips:
            upload_files.extend(_split_oversized_file(zp, out_dir))

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        total = len(upload_files)
        for i, fpath in enumerate(upload_files, start=1):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            part_label = f" (Part {i}/{total})" if total > 1 else ""
            caption = f"📦 {reason}{part_label}\n🕒 {tag}\n💾 Size: {size_mb:.2f} MB ({len(staged)} files)"
            with open(fpath, "rb") as fh:
                resp = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                    files={"document": fh},
                    timeout=120,
                )
            resp.raise_for_status()

        set_config("last_backup_ts", str(time.time()))
        return True, f"Delivered {total} backup file(s) covering {len(staged)} data files."
    except Exception as e:
        return False, f"Backup failed: {e}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def start_backup_scheduler(interval_days: int = DEFAULT_BACKUP_INTERVAL_DAYS) -> None:
    """
    Starts a lightweight background thread that ensures an automatic Telegram backup
    is delivered every `interval_days` days (default: 30 days), persisted in SQLite.
    """
    if interval_days <= 0:
        return

    # Initialize timestamp on first boot so backup triggers cleanly after the first interval
    if not get_config("last_backup_ts"):
        set_config("last_backup_ts", str(time.time()))

    interval_seconds = interval_days * 86400

    def _loop() -> None:
        while True:
            try:
                last_raw = get_config("last_backup_ts", "0")
                last_ts = float(last_raw) if last_raw else 0.0
                if time.time() - last_ts >= interval_seconds:
                    ok, msg = send_telegram_backup(reason=f"Auto {interval_days}-Day Backup")
                    print(f"[Backup Scheduler] ok={ok}: {msg}")
            except Exception as e:
                print(f"[Backup Scheduler] Error: {e}")
            time.sleep(6 * 3600)  # Check every 6 hours

    threading.Thread(target=_loop, daemon=True, name="backup-scheduler").start()
