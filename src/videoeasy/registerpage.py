"""A local showcase of the register map: how the run went, its most confident tags, and where its readers disagree.

`register showcase` writes `work/reviews/register-showcase-<date>/index.html` with a close-up clip per moment
(cropped around the face track, so the expression is visible on a phone-sized card), the boxed context frame
and the crops the vision model was shown. Confidence is chosen in code (CONFIDENT), never by a model: the
moment is `agreed`, read at strength 2 or more, silent, and the face is large enough to see. It is a picture of
the uncalibrated system for the editor, not a result. Like the review page it is never published: it shows
people's faces.
"""
from __future__ import annotations

import datetime as dt
import html
import io
import json
import subprocess
import sys
from pathlib import Path

from . import faces as faces_mod
from .frames import _escape_filter_path
from .register import CROP_PX, fmt_t, frames_for

PER_REGISTER = 6
MIN_PX = 80              # a confident example must be a face you can see on a card
CLOSE_PX = 360           # close-up clip width
LEAD_S = 0.8             # the clip opens this long before the moment


def confidence(m: dict) -> float:
    """Rank of an agreed moment: strength first, then how many readings back it, how far the backing cue rose over
    the face's own rest, the face's size, and whether its baseline is its own."""
    f, r = m["fused"], m["reading"]
    backing = {s for s in f.get("support", []) if not s.startswith("heard:")}
    rise = max((c["peak"] - c["baseline"] for c in m["cues"]
                if c["cue"] in backing and c["peak"] is not None and c["baseline"] is not None), default=0.0)
    return r["strength"] * 10 + len(f.get("support", [])) * 2 + rise * 5 + min(m.get("px") or 0, 300) / 100 + (m.get("baseline") == "own")


def confident(m: dict) -> bool:
    f, r = m["fused"], m.get("reading") or {}
    return (f["status"] == "agreed" and r.get("strength", 0) >= 2 and not f["speaking"]
            and (m["track"] is None or (m.get("px") or 0) >= MIN_PX))


def pick(moments: list[dict], n: int, key) -> list[dict]:
    """The top n by key, at most one per source, so a page of examples is not one clip n times."""
    out, seen = [], set()
    for m in sorted(moments, key=key, reverse=True):
        if m["source"] in seen:
            continue
        out.append(m)
        seen.add(m["source"])
        if len(out) == n:
            break
    return out


def select(doc: dict, per: int = PER_REGISTER) -> dict:
    ms = doc["moments"]
    by_reg = {t: pick([m for m in ms if confident(m) and m["fused"]["label"] == t], per, confidence) for t in doc["vocabulary"]}
    heard = pick([m for m in ms if confident(m) and any(s.startswith("heard:") for s in m["fused"].get("support", []))], 4, confidence)
    seen_only = pick([m for m in ms if m["fused"]["status"] == "seen_only" and m["reading"]["strength"] >= 3 and not m["fused"]["speaking"]
                      and (m.get("px") or 0) >= MIN_PX], 4, lambda m: m.get("px") or 0)
    measured_only = pick([m for m in ms if m["fused"]["status"] == "measured_only" and (m.get("px") or 0) >= MIN_PX and m["cues"]],
                         4, lambda m: max((c["peak"] or 0) - (c["baseline"] or 0) for c in m["cues"]))
    return dict(by_register=by_reg, heard=heard, seen_only=seen_only, measured_only=measured_only)


def _source_size(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def closeup_box(m: dict, rec: dict, src_wh: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A square crop in SOURCE pixels around everywhere the track's face was during the clip, padded."""
    tr = next((t for t in rec.get("tracks", []) if t["id"] == m["track"]), None)
    if tr is None or not rec.get("frame_w"):
        return None
    a, b = m["t0"] - LEAD_S, m["t1"] + LEAD_S
    boxes = [s[1:5] for s in tr["samples"] if a <= s[0] <= b] or [s[1:5] for s in tr["samples"]]
    k = src_wh[0] / rec["frame_w"]
    x0 = min(x for x, _, _, _ in boxes) * k
    y0 = min(y for _, y, _, _ in boxes) * k
    x1 = max(x + w for x, _, w, _ in boxes) * k
    y1 = max(y + h for _, y, _, h in boxes) * k
    side = min(max(x1 - x0, y1 - y0) * 2.2, *src_wh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(0, cx - side / 2), src_wh[0] - side)
    top = min(max(0, cy - side / 2), src_wh[1] - side)
    even = lambda v: int(v) // 2 * 2  # noqa: E731
    return even(side), even(side), even(left), even(top)


def media(m: dict, rec: dict, root: Path, cfg) -> dict:
    """Close-up clip, context frame and the model's crops for one moment (cached by moment id)."""
    import hashlib
    from PIL import Image
    slug = hashlib.sha256(m["id"].encode()).hexdigest()[:12]
    clip, ctx, strip = root / f"{slug}.mp4", root / f"{slug}-ctx.jpg", root / f"{slug}-strip.jpg"
    grade = cfg.grade_for(m["path"])
    pre = ([f"lut3d={_escape_filter_path(grade.lut)}"] if grade.lut else []) + ([grade.filters] if grade.filters else [])
    if not clip.exists():
        box = closeup_box(m, rec, _source_size(m["path"])) if m["track"] else None
        vf = pre + ([f"crop={box[0]}:{box[1]}:{box[2]}:{box[3]}", f"scale={CLOSE_PX}:-2"] if box else ["scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)'"])
        a = max(0.0, m["t0"] - LEAD_S)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{a:.2f}", "-i", m["path"],
                        "-t", f"{m['t1'] - m['t0'] + 2 * LEAD_S:.2f}", "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "24",
                        "-preset", "veryfast", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(clip)], capture_output=True)
    if not ctx.exists():
        try:
            imgs, _ = frames_for(m, rec, grade)
            Image.open(io.BytesIO(imgs[0])).save(ctx, quality=85)
            tiles = [Image.open(io.BytesIO(b)) for b in imgs[1:]] if m["track"] else []
            if tiles:
                t = CROP_PX // 3
                sheet = Image.new("RGB", (t * len(tiles), t))
                for k, im in enumerate(tiles):
                    sheet.paste(im.resize((t, t)), (k * t, 0))
                sheet.save(strip, quality=85)
        except Exception as e:  # noqa: BLE001
            print(f"  frames for {m['id']}: {e}", file=sys.stderr)
    return dict(clip=clip.name if clip.exists() and clip.stat().st_size > 2000 else None,
                ctx=ctx.name if ctx.exists() else None, strip=strip.name if strip.exists() else None)


def build(cfg, per: int = PER_REGISTER) -> Path:
    doc = json.loads((cfg.out_dir / "register.json").read_text())
    chosen = select(doc, per)
    root = cfg.work_dir / "reviews" / f"register-showcase-{dt.date.today():%Y%m%d}"
    (root / "media").mkdir(parents=True, exist_ok=True)
    recs: dict[str, dict] = {}
    cards: dict[str, dict] = {}
    todo = [m for ms in chosen["by_register"].values() for m in ms] + chosen["heard"] + chosen["seen_only"] + chosen["measured_only"]
    for i, m in enumerate(todo, 1):
        if m["id"] in cards:
            continue
        rec = recs.setdefault(m["source"], faces_mod.load_source(cfg.out_dir, m["source"]) or dict(tracks=[]))
        cards[m["id"]] = media(m, rec, root / "media", cfg)
        print(f"[{i}/{len(todo)}] {m['id']}", file=sys.stderr)
    page = root / "index.html"
    page.write_text(page_html(doc, chosen, cards, per))
    return page


# ----------------------------------------------------------------------- page
def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _bars(rows: list[tuple[str, int, str]], total: int | None = None) -> str:
    top = max([v for _, v, _ in rows] or [1]) or 1
    out = []
    for label, v, note in rows:
        pct = 100 * v / top
        share = f" · {100 * v / total:.0f}%" if total else ""
        out.append(f'<div class="bar" title="{_e(label)}: {v}{share}"><span class="bl">{_e(label)}</span>'
                   f'<span class="track"><span class="fill" style="width:{pct:.1f}%"></span></span>'
                   f'<span class="bv">{v}{_e(share)}</span>{f"<span class=bn>{_e(note)}</span>" if note else ""}</div>')
    return '<div class="bars">' + "".join(out) + "</div>"


def _dots(n: int) -> str:
    return '<span class="dots" aria-label="strength %d of 3">%s</span>' % (n, "".join("●" if k < n else "○" for k in range(3)))


def card(m: dict, med: dict, note: str | None = None) -> str:
    f, r = m["fused"], m.get("reading") or {}
    reg = f.get("label") or r.get("register") or "none"
    cues = "".join(f'<li><b>{_e(c["cue"].replace("_", " "))}</b> {c["peak"]:.2f}'
                   + (f' <span class="mut">rest {c["baseline"]:.2f}</span>' if c.get("baseline") is not None else "") + "</li>"
                   for c in m["cues"] if c["peak"] is not None)
    if any(c["cue"] == "still" for c in m["cues"]):
        cues += "<li><b>still face</b> <span class=mut>no cue up, lips still</span></li>"
    heard = "".join(f'<li><b>{_e(h["family"])}</b> {h["peak"]:.2f}</li>' for h in m["heard"])
    video = (f'<video src="media/{_e(med["clip"])}" muted loop playsinline preload="metadata" controls></video>' if med.get("clip")
             else '<div class="novid">no clip</div>')
    face = f'{m["px"]:.0f} px face · {"own" if m.get("baseline") == "own" else "population"} baseline' if m.get("track") else "sound-led: anyone visible"
    model_saw = ""
    if med.get("ctx") or med.get("strip"):
        model_saw = ('<details><summary>What the model was shown</summary>'
                     + (f'<img src="media/{_e(med["ctx"])}" alt="full frame, face boxed" loading="lazy">' if med.get("ctx") else "")
                     + (f'<img src="media/{_e(med["strip"])}" alt="face crops in time order" loading="lazy">' if med.get("strip") else "")
                     + "</details>")
    return f"""<article class="card">{video}
<div class="body"><div class="head"><span class="chip">{_e(reg)}</span>{_dots(r.get("strength", 0))}<span class="where">{_e(m["source"])} · {fmt_t(m["t0"])}–{fmt_t(m["t1"])}</span></div>
{f'<p class="note">{_e(note)}</p>' if note else ""}
<p class="seen">“{_e(r.get("visible", ""))}”</p>
<div class="ev"><div><h4>Measured</h4><ul>{cues or "<li class=mut>no cue</li>"}</ul></div><div><h4>Heard</h4><ul>{heard or "<li class=mut>nothing</li>"}</ul></div></div>
<p class="meta">{_e(face)}{" · tears seen" if r.get("tears") == "yes" else ""}</p>{model_saw}</div></article>"""


def page_html(doc: dict, chosen: dict, cards: dict[str, dict], per: int) -> str:
    c, st, ms = doc["coverage"], doc["status"], doc["moments"]
    read = sum(v for k, v in st.items() if k not in ("unchecked",))
    failed = st.get("invalid", 0) + st.get("unchecked", 0)
    strong = {t: sum(1 for m in ms if confident(m) and m["fused"]["label"] == t) for t in doc["vocabulary"]}
    status_rows = [("agreed: seen and backed by a measurement or sound", st.get("agreed", 0), ""),
                   ("seen only: the model named one, nothing backs it", st.get("seen_only", 0), ""),
                   ("measured only: muscles moved, the model saw nothing", st.get("measured_only", 0), ""),
                   ("none: a quiet face, and the model agreed", st.get("none", 0), ""),
                   ("unreadable: too small, turned away or covered", st.get("unreadable", 0), "")]
    reg_rows = sorted(((t, doc["labels"].get(t, 0), f"{strong[t]} strong & silent") for t in doc["vocabulary"]), key=lambda x: -x[1])
    sections = []
    for t, d in doc["vocabulary"].items():
        hits = chosen["by_register"][t]
        body = "".join(card(m, cards[m["id"]]) for m in hits) if hits else (
            '<p class="empty">No confident example. ' + ("The model never named it on these clips."
                                                         if not any((m.get("reading") or {}).get("register") == t for m in ms)
                                                         else "It was named, but never at strength 2+ with backing on a silent, visible face.") + "</p>")
        sections.append(f'<section id="{_e(t)}"><h2>{_e(t)} <span class="count">{doc["labels"].get(t, 0)} agreed · {strong[t]} strong &amp; silent</span></h2>'
                        f'<p class="def">Definition the model was given: <i>{_e(d)}</i></p><div class="grid">{body}</div></section>')
    heard = "".join(card(m, cards[m["id"]]) for m in chosen["heard"])
    so = "".join(card(m, cards[m["id"]], "The model was sure (strength 3); no measured cue or sound backs it.") for m in chosen["seen_only"])
    mo = "".join(card(m, cards[m["id"]], "The strongest cue rises in the run; the model saw no expression.") for m in chosen["measured_only"])
    nav = " · ".join(f'<a href="#{_e(t)}">{_e(t)}</a>' for t in doc["vocabulary"]) + ' · <a href="#heard">heard</a> · <a href="#disagree">disagreements</a>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Register map showcase</title><style>
:root{{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--fg:#0b0b0b;--sec:#52514e;--mut:#8a887f;--line:#e6e4dd;--track:#efede7;--s1:#2a78d6;--chip:#eef4fc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#ffffff;--sec:#c3c2b7;--mut:#8f8d84;--line:#34332f;--track:#2c2b28;--s1:#3987e5;--chip:#1f2c3d}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#ffffff;--sec:#c3c2b7;--mut:#8f8d84;--line:#34332f;--track:#2c2b28;--s1:#3987e5;--chip:#1f2c3d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:20px;margin:40px 0 4px;text-transform:capitalize}}
h2 .count{{font-size:14px;font-weight:400;color:var(--sec);text-transform:none;margin-left:8px}}h3{{font-size:15px;margin:0 0 10px}}
.lede{{color:var(--sec);max-width:760px;margin:0 0 18px}}nav{{font-size:14px;color:var(--mut);margin:0 0 24px}}nav a{{color:var(--sec)}}
.caveat{{border-left:3px solid var(--s1);padding:8px 14px;background:var(--card);color:var(--sec);max-width:820px;margin:0 0 24px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:0 0 24px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}.tile b{{display:block;font-size:26px;font-weight:600}}.tile span{{color:var(--sec);font-size:13px}}
.panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.bars{{display:grid;gap:8px}}.bar{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(60px,1fr) auto;gap:4px 10px;align-items:center;font-size:13px}}
.bl{{color:var(--sec)}}.track{{height:12px;background:var(--track);border-radius:0 4px 4px 0}}.fill{{display:block;height:12px;background:var(--s1);border-radius:0 4px 4px 0;min-width:2px}}
.bv{{font-variant-numeric:tabular-nums;color:var(--fg)}}.bn{{grid-column:1/-1;color:var(--mut);font-size:12px;margin-top:-4px}}
.def{{color:var(--sec);font-size:14px;margin:0 0 14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,270px),1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}}
.card video{{width:100%;aspect-ratio:1/1;object-fit:cover;background:#000;display:block}}.novid{{aspect-ratio:1/1;display:grid;place-items:center;color:var(--mut)}}
.body{{padding:10px 12px 12px}}.head{{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center}}.chip{{background:var(--chip);border-radius:999px;padding:1px 10px;font-size:13px;font-weight:600;text-transform:capitalize}}
.dots{{color:var(--s1);letter-spacing:1px;font-size:12px}}.where{{color:var(--mut);font-size:12px;width:100%;word-break:break-all}}
.seen{{margin:8px 0;font-size:14px}}.note{{margin:8px 0 0;font-size:13px;color:var(--sec)}}.ev{{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px}}
.ev h4{{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}}.ev ul{{margin:2px 0 0;padding:0;list-style:none}}.mut{{color:var(--mut)}}
.meta{{color:var(--mut);font-size:12px;margin:8px 0 0}}details{{margin-top:8px;font-size:12px;color:var(--sec)}}details img{{width:100%;margin-top:6px;border-radius:6px;display:block}}
.empty{{color:var(--sec);font-style:italic}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:3px 10px 3px 0;text-align:left}}td.n{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body><main>
<h1>Register map: the confident end</h1>
<p class="lede">Every face in {c["measured"]} clips was measured, every candidate moment was read by the local vision model, and the two were fused in code. Below are the moments the system is surest about for each register, then the places where its readers disagree.</p>
<div class="caveat"><b>Not calibrated yet.</b> Confidence here is the system agreeing with itself: the model read the face at strength 2 or 3, a measured cue or a sound backs the reading, the lips are still, and the face is at least {MIN_PX} px. Your blind labels on the review page decide how far each register can be trusted. A tag describes what a face shows, never what a man feels.</div>
<nav>{nav}</nav>
<div class="tiles"><div class="tile"><b>{c["footage_s"] / 60:.0f} min</b><span>footage measured, {c["measured"]} of {c["sources"]} clips</span></div>
<div class="tile"><b>{c["sources_with_readable_face"]}</b><span>clips with at least one readable face</span></div>
<div class="tile"><b>{read:,}</b><span>moments read by the vision model, {failed} failed</span></div>
<div class="tile"><b>{st.get("agreed", 0)}</b><span>tags where seen and measured agree</span></div>
<div class="tile"><b>{sum(strong.values())}</b><span>of those strong (2+), silent and visible</span></div></div>
<div class="panels"><div class="panel"><h3>What the three readers concluded, per moment</h3>{_bars(status_rows, read)}</div>
<div class="panel"><h3>Agreed tags per register</h3>{_bars(reg_rows)}</div></div>
<details style="margin-top:12px"><summary>Same numbers as a table</summary><table><tr><th>outcome</th><th>moments</th></tr>{"".join(f"<tr><td>{_e(k)}</td><td class=n>{v}</td></tr>" for k, v in sorted(st.items(), key=lambda x: -x[1]))}</table>
<table style="margin-top:8px"><tr><th>register</th><th>agreed</th><th>strong &amp; silent</th></tr>{"".join(f"<tr><td>{_e(t)}</td><td class=n>{doc['labels'].get(t, 0)}</td><td class=n>{strong[t]}</td></tr>" for t in doc["vocabulary"])}</table></details>
{"".join(sections)}
<section id="heard"><h2>Backed by sound</h2><p class="def">Agreed moments where the clip's own audio (laughter, cheering, a shout) supports the reading. The sound is never pinned to the face: it may come from off camera.</p><div class="grid">{heard or '<p class="empty">None.</p>'}</div></section>
<section id="disagree"><h2>Where the readers disagree</h2><p class="def">The model alone, very sure:</p><div class="grid">{so or '<p class="empty">None.</p>'}</div>
<p class="def" style="margin-top:20px">The measurement alone: the strongest cue rises the model read as no expression:</p><div class="grid">{mo or '<p class="empty">None.</p>'}</div></section>
<p class="meta" style="margin-top:40px">Built {dt.datetime.now():%d %b %Y %H:%M} from register.json ({_e(doc["created"])}). Local file: it shows participants' faces and is never published.</p>
</main><script>
const io=new IntersectionObserver(es=>es.forEach(e=>{{const v=e.target;if(e.isIntersecting){{v.play().catch(()=>{{}})}}else{{v.pause()}}}}),{{threshold:.5}});
document.querySelectorAll("video").forEach(v=>io.observe(v));
</script></body></html>"""
