"""Small, atomic creative records; audio and codec artifacts stay in their files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from threading import RLock
import time
import uuid


def record_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Invalid studio record ID")
    return value


def clone(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class StudioStore:
    def __init__(self, root):
        self.path = Path(root) / "studio.json"
        self.lock = RLock()
        self.data = {"schema": 1, "drafts": {}, "works": {}}
        if self.path.is_file():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("schema") != 1 or not isinstance(value.get("works"), dict):
                raise ValueError("Unsupported studio document")
            self.data = value

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(
                json.dumps(self.data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self):
        with self.lock:
            return clone(self.data)

    def save_draft(self, key, value):
        record_id(key)
        if not isinstance(value, dict) or len(json.dumps(value, ensure_ascii=False)) > 2 * 1024**2:
            raise ValueError("Studio draft must be an object no larger than 2 MiB")
        with self.lock:
            self.data["drafts"][key] = {"updated_at": time.time(), "value": clone(value)}
            self._save()
            return clone(self.data["drafts"][key])

    def removed(self, kind):
        with self.lock:
            return set(self.data.get("removed", {}).get(kind, []))

    def remove_from_list(self, kind, identifiers):
        if not identifiers:
            return
        with self.lock:
            before = clone(self.data)
            try:
                removed = self.data.setdefault("removed", {})
                removed[kind] = sorted(self.removed(kind) | set(identifiers))
                if kind == "jobs":
                    for draft in self.data["drafts"].values():
                        self.clear_results(draft["value"], identifiers.__contains__)
                self._save()
            except OSError:
                self.data = before
                raise

    @staticmethod
    def clear_results(value, is_removed):
        if isinstance(value, dict):
            for field, child in value.items():
                if field in ("active_job_id", "result_job_id", "resultJobId"):
                    if isinstance(child, str) and is_removed(child):
                        value[field] = None
                elif field == "active_result_source" and isinstance(child, dict):
                    if isinstance(child.get("job_id"), str) and is_removed(child["job_id"]):
                        value[field] = None
                else:
                    StudioStore.clear_results(child, is_removed)
        elif isinstance(value, list):
            for child in value:
                StudioStore.clear_results(child, is_removed)

    def forget_removed(self, jobs, voices):
        if not jobs and not voices:
            return
        with self.lock:
            before = clone(self.data)
            try:
                self.data["removed"].update(jobs=sorted(self.removed("jobs") - jobs),
                                            voices=sorted(self.removed("voices") - voices))
                works = list(self.data["works"].items()) if jobs else []
                for work_id, work in works:
                    for part in work["segments"]:
                        part["versions"] = [identifier for identifier in part["versions"] if identifier not in jobs]
                        for key in ("adopted", "scene_version"):
                            if (part.get(key) or {}).get("job_id") in jobs:
                                part[key] = None
                        if "line_versions" in part:
                            part["line_versions"] = {key: value for key, value in part["line_versions"].items()
                                                     if value.get("job_id") not in jobs}
                    work["segments"] = [part for part in work["segments"] if part["versions"]]
                    if not work["segments"]:
                        del self.data["works"][work_id]
                self._save()
            except OSError:
                self.data = before
                raise

    def register(self, job):
        context = job.get("studio")
        if not context:
            return
        work_id, segment_id = record_id(context["work_id"]), record_id(context["segment_id"])
        with self.lock:
            work = self.data["works"].setdefault(work_id, {
                "id": work_id, "created_at": time.time(), "segments": [],
                "layout": context.get("layout", "single"), "title": job["recipe"].get("text", "")[:80],
            })
            segment = next((row for row in work["segments"] if row["id"] == segment_id), None)
            if segment is None:
                segment = {"id": segment_id, "versions": [], "adopted": None}
                work["segments"].append(segment)
            segment["versions"].append(job["id"])
            if job["recipe"].get("scene_order"):
                segment["line_order"] = clone(job["recipe"]["scene_order"])
            elif job["recipe"].get("mode") == "scene" and job["recipe"].get("line_ids"):
                segment.setdefault("line_order", clone(job["recipe"]["line_ids"]))
            if (job["recipe"].get("mode") == "scene"
                    and job["recipe"].get("scene_strategy", "per_line") == "per_line"
                    and not context.get("line_id") and not segment.get("scene_version")):
                segment["scene_version"] = {"job_id": job["id"]}
            work["updated_at"] = time.time()
            self._save()

    def first_completed(self, job):
        context = job.get("studio")
        if not context or job.get("status") != "complete" or not job.get("output"):
            return
        with self.lock:
            work = self.data["works"].get(context["work_id"])
            if work is None:
                return
            segment = next(row for row in work["segments"] if row["id"] == context["segment_id"])
            if context.get("line_id") and not job["recipe"].get("source_version"):
                segment.setdefault("line_versions", {}).setdefault(
                    record_id(context["line_id"]), {"job_id": job["id"]})
                work["updated_at"] = time.time()
                self._save()
            elif not context.get("line_id") and segment["adopted"] is None:
                segment["adopted"] = {"job_id": job["id"]}
                work["updated_at"] = time.time()
                self._save()

    def adopt(self, work_id, segment_id, version):
        with self.lock:
            work = self.data["works"].get(record_id(work_id))
            if work is None:
                raise ValueError("Studio work does not exist")
            segment = next((row for row in work["segments"] if row["id"] == record_id(segment_id)), None)
            if segment is None or version["job_id"] not in segment["versions"]:
                raise ValueError("Version does not belong to this segment")
            if segment.get("adopted") != version:
                segment.pop("line_versions", None)
            segment["adopted"] = clone(version)
            work["updated_at"] = time.time()
            self._save()
            return clone(work)

    def adopt_line(self, work_id, segment_id, line_id, version):
        with self.lock:
            work = self.data["works"].get(record_id(work_id))
            segment = next((row for row in (work or {}).get("segments", [])
                            if row["id"] == record_id(segment_id)), None)
            if segment is None or version["job_id"] not in segment["versions"]:
                raise ValueError("Studio segment does not exist")
            segment.setdefault("line_versions", {})[record_id(line_id)] = clone(version)
            work["updated_at"] = time.time()
            self._save()
            return clone(work)
