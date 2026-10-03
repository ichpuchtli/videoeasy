"""Stage 7: extract marked passages + comments from the transcript .docx.

Google Docs exports carry editorial markup in three forms, all of which are
prior human judgment for the beat map:
  - true highlights   -> w:highlight (named colors)
  - text background   -> w:shd run shading (hex fills)  <- Docs' background-color tool
  - comments          -> word/comments.xml + inline anchor ranges

python-docx doesn't surface shading or comments, so this parses the XML.
"""
from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET

from .config import Config

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# friendly names for the fills seen in Google Docs exports; unknown hexes pass through
FILL_NAMES = {
    "fff2cc": "pale-yellow",
    "ffe599": "light-yellow",
    "ff9900": "orange",
    "ead1dc": "pink",
    "7f6000": "dark-olive",
}


def _run_marker(run: ET.Element) -> str | None:
    rpr = run.find(f"{W}rPr")
    if rpr is None:
        return None
    highlight = rpr.find(f"{W}highlight")
    if highlight is not None:
        return highlight.get(f"{W}val")
    shd = rpr.find(f"{W}shd")
    if shd is not None:
        fill = (shd.get(f"{W}fill") or "").lower()
        if fill not in ("", "auto", "ffffff"):
            return FILL_NAMES.get(fill, f"#{fill}")
    return None


def _run_text(run: ET.Element) -> str:
    return "".join(t.text or "" for t in run.iter(f"{W}t"))


def extract_marks(root: ET.Element) -> list[dict]:
    """Contiguous same-marker runs merged, per paragraph."""
    marks: list[dict] = []
    for para in root.iter(f"{W}p"):
        current_text: list[str] = []
        current_marker: str | None = None

        def flush():
            nonlocal current_text, current_marker
            text = "".join(current_text).strip()
            if text and current_marker:
                marks.append({"text": text, "color": current_marker})
            current_text, current_marker = [], None

        for run in para.iter(f"{W}r"):
            marker = _run_marker(run)
            if marker != current_marker:
                flush()
                current_marker = marker
            if marker:
                current_text.append(_run_text(run))
        flush()
    return marks


def extract_comments(docx_path: str) -> list[dict]:
    with zipfile.ZipFile(docx_path) as z:
        doc_root = ET.fromstring(z.read("word/document.xml"))
        try:
            comments_root = ET.fromstring(z.read("word/comments.xml"))
        except KeyError:
            return []

    # anchors: walk document order, capture covered text per comment id
    anchors: dict[str, str] = {}
    open_ids: dict[str, list[str]] = {}
    for el in doc_root.iter():
        if el.tag == f"{W}commentRangeStart":
            open_ids[el.get(f"{W}id")] = []
        elif el.tag == f"{W}commentRangeEnd":
            cid = el.get(f"{W}id")
            if cid in open_ids:
                anchors[cid] = "".join(open_ids.pop(cid)).strip()
        elif el.tag == f"{W}t" and open_ids:
            for buf in open_ids.values():
                buf.append(el.text or "")

    # Google exports duplicate comments (sometimes with different anchor spans);
    # dedupe on (author, text), keeping the longest anchor.
    deduped: dict[tuple, dict] = {}
    for c in comments_root.iter(f"{W}comment"):
        cid = c.get(f"{W}id")
        text = " ".join(
            "".join(t.text or "" for t in p.iter(f"{W}t")) for p in c.iter(f"{W}p")
        ).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        entry = {
            "author": c.get(f"{W}author", ""),
            "comment": text,
            "anchor": anchors.get(cid, ""),
        }
        key = (entry["author"], entry["comment"])
        if key not in deduped or len(entry["anchor"]) > len(deduped[key]["anchor"]):
            deduped[key] = entry
    return list(deduped.values())


def run(cfg: Config) -> dict:
    if not cfg.selects_docx.is_file():
        print(f"no docx at {cfg.selects_docx} — skipping selects")
        return {}
    with zipfile.ZipFile(cfg.selects_docx) as z:
        doc_root = ET.fromstring(z.read("word/document.xml"))
    result = {
        "marks": extract_marks(doc_root),
        "comments": extract_comments(str(cfg.selects_docx)),
    }
    (cfg.out_dir / "selects.json").write_text(json.dumps(result, indent=2))
    by_color: dict[str, int] = {}
    for m in result["marks"]:
        by_color[m["color"]] = by_color.get(m["color"], 0) + 1
    print(f"{len(result['marks'])} marked passages {by_color}, {len(result['comments'])} comments")
    return result
