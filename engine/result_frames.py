"""Parse and deterministically bound compact child-to-parent result frames."""

from __future__ import annotations

import json
import re
from typing import Any

from engine.context_governor import count_text_tokens
from engine.interfaces import EvidenceRef, ResultFrame, RiskItem


def _clip(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    marker = "...[trimmed]"
    return text[:max(0, limit - len(marker))] + marker


def _strings(value: object, limit: int, item_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clip(item, item_chars) for item in value[:limit] if str(item).strip()]


def _line(value: object) -> int | None:
    try:
        number = int(value)
        return number if number >= 1 else None
    except (TypeError, ValueError):
        return None


def _candidate_objects(text: str):
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            yield parsed
    except (json.JSONDecodeError, TypeError):
        pass

    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                yield parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            if isinstance(parsed, dict):
                yield parsed
        except json.JSONDecodeError:
            continue


def _find_frame_data(text: str) -> dict[str, Any] | None:
    for candidate in _candidate_objects(text):
        nested = candidate.get("result_frame")
        if isinstance(nested, dict):
            candidate = nested
        if "summary" in candidate and any(
            key in candidate for key in (
                "status", "evidence", "changes_needed", "risks",
                "open_questions", "confidence")
        ):
            return candidate
    return None


def _bounded_frame(frame: ResultFrame, max_tokens: int) -> ResultFrame:
    """Remove optional collection tails before truncating the summary."""
    while count_text_tokens(frame.model_dump_json()) > max_tokens:
        if frame.open_questions:
            frame.open_questions.pop()
        elif frame.risks:
            frame.risks.pop()
        elif frame.changes_needed:
            frame.changes_needed.pop()
        elif frame.evidence:
            frame.evidence.pop()
        else:
            break

    if count_text_tokens(frame.model_dump_json()) <= max_tokens:
        return frame

    original = frame.summary
    low, high = 0, len(original)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = frame.model_copy(update={"summary": original[:middle]})
        if count_text_tokens(candidate.model_dump_json()) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    frame.summary = original[:low] if low else ""
    return frame


def build_result_frame(
    raw_result: str,
    node_id: str,
    success: bool,
    config: dict[str, Any] | None = None,
) -> ResultFrame:
    """Parse a model frame or construct a bounded prose fallback."""
    settings = (config or {}).get("result_frames", {})
    max_tokens = int(settings.get("max_tokens", 800))
    summary_chars = int(settings.get("max_summary_chars", 1800))
    evidence_limit = int(settings.get("max_evidence", 6))
    finding_chars = int(settings.get("max_finding_chars", 500))
    changes_limit = int(settings.get("max_changes", 6))
    risks_limit = int(settings.get("max_risks", 4))
    questions_limit = int(settings.get("max_questions", 4))
    item_chars = int(settings.get("max_item_chars", 400))
    data = _find_frame_data(raw_result) or {}

    evidence = []
    raw_evidence = data.get("evidence", [])
    if isinstance(raw_evidence, list):
        for item in raw_evidence[:evidence_limit]:
            if not isinstance(item, dict):
                continue
            evidence.append(EvidenceRef(
                file_path=_clip(item.get("file_path", ""), item_chars),
                symbol_name=_clip(item.get("symbol_name"), item_chars) or None,
                line_start=_line(item.get("line_start")),
                line_end=_line(item.get("line_end")),
                finding=_clip(item.get("finding", ""), finding_chars),
            ))

    risks = []
    raw_risks = data.get("risks", [])
    if isinstance(raw_risks, list):
        for item in raw_risks[:risks_limit]:
            if isinstance(item, dict):
                description = item.get("description", "")
                severity = item.get("severity", "unknown")
            else:
                description, severity = item, "unknown"
            if str(description).strip():
                risks.append(RiskItem(
                    description=_clip(description, item_chars),
                    severity=_clip(severity, 40) or "unknown"))

    confidence = data.get("confidence")
    try:
        confidence = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None

    summary = data.get("summary") if data else raw_result
    frame = ResultFrame(
        node_id=node_id,
        status="success" if success else "failed",
        summary=_clip(summary or raw_result, summary_chars),
        evidence=evidence,
        changes_needed=_strings(data.get("changes_needed"), changes_limit, item_chars),
        risks=risks,
        open_questions=_strings(data.get("open_questions"), questions_limit, item_chars),
        confidence=confidence,
    )
    return _bounded_frame(frame, max_tokens)
