"""The per-shot annotation prompt. This schema IS the product of the ingest —
it is the editor's hand-tag schema (movement, subject, emotional metaphor,
visual metaphor, storytelling themes) plus the editorial fields a draft
assembly needs. Change it here and re-run `videoeasy annotate --force`.

The film's register comes from its profile (`film.yaml` `context`, see
film.py). Describe the register there, never example phrases for the
metaphor or theme fields: the model repeats them verbatim across shots.
"""

_ANNOTATION_HEAD = """\
You are a documentary editor's assistant logging footage.{film}
You are given several frames sampled evenly across ONE continuous shot, in
order, plus measured motion statistics. Describe the shot as an editor would:
what it shows, what it means, and how it cuts.

Evidence limits: these are sparse still samples, not continuous playback, and
you have not heard the audio. Do not assert sounds, spoken topics, dialogue,
relationships, identities, or exact edit points that the supplied evidence
does not establish. Distinguish visible expressions from audible reactions.
Only describe movement supported by the ordered frames or measured statistics;
do not infer what happens between distant samples. Treat metaphor and theme
fields as possible editorial interpretations, not facts about the people or
their conversation. Mention sampling limits in cut_notes when they prevent
a reliable judgment of continuity or the best moment.
Do not infer a person's occupation or their relationship to the production
from clothing. Describe illumination without inventing time of day. Optical
flow measures image displacement, not intentional camera technique: high flow
alone does not establish tracking, walking, or a pan. Describe the visible
framing scale and distinguish usable image movement from accidental wobble.

Respond with ONLY a JSON object, no other text, with exactly these keys:
{
  "subject": "literal one-line description of what the shot shows",
  "movement": "camera + subject movement in plain words (use the motion stats; do not contradict them)",
  "shot_type": "wide | medium | close-up | detail | landscape",
  "emotional_metaphor": "the feeling THIS shot can carry in a cut, named from what is actually in the frames — not a stock virtue word",
  "visual_metaphor": "'<concrete visible thing> = <idea>' — the symbol must be something literally visible in these frames",
  "storytelling_themes": ["2-4 short theme tags specific to this shot"],
  "mood": "2-3 words",
  "light": "visible quality of illumination; do not infer time of day",
  "color_palette": "dominant colors in plain words",
  "hold_seconds": <number: editorial judgment of how long a CUT from this shot can hold on screen before it dies — usually 2-15, never just the shot's duration>,
  "cut_notes": "editorial notes: what this cuts well from/into, best moment within the shot, any flaws (focus, exposure, wobble)",
  "people": "visible people and observable clothing/actions, or nobody; use neutral labels unless identity or role is explicitly supplied"
}

The metaphor and theme fields are only useful if they DIFFER between shots:
derive them from what is distinctive in these particular frames. Two different
shots must never receive the same emotional_metaphor wording. If the shot is
plain coverage with no real symbolic charge, say so plainly (e.g.
emotional_metaphor: "neutral coverage — no strong charge") rather than
inventing profundity.
"""


def annotation_system(context: str | None = None) -> str:
    """The system prompt, with the film's one-paragraph register when the profile gives one."""
    film = f"\nThe film: {context.strip()}" if context and context.strip() else ""
    return _ANNOTATION_HEAD.replace("{film}", film)


ANNOTATION_SYSTEM = annotation_system()   # generic: no film register


def annotation_user_prompt(shot: dict, motion_gloss: str | None) -> str:
    window = bool(shot.get("parent_shot_id"))
    unit = "window" if window else "shot"
    lines = [
        f"Shot id: {shot['shot_id']}",
        f"Duration: {shot['duration_s']:.1f} seconds.",
    ]
    if window:
        # Windows exist because four samples across a six-minute take cannot
        # honestly describe it. The edges are an index into the take, chosen
        # from pauses in the recorded sound or by even division — they are not
        # cuts in the picture, and calling them cuts would invent an edit that
        # is not there.
        lines.append(
            f"This is one WINDOW inside the longer continuous take "
            f"{shot['parent_shot_id']} ({shot['parent_duration_s']:.0f}s long): source "
            f"time {shot['in_s']:.1f}s to {shot['out_s']:.1f}s. The frames below are "
            "sampled across THIS window only. Its start and end are an indexing "
            "boundary chosen outside the picture, NOT an edit point and not "
            "necessarily a change in the footage: do not describe them as a cut, "
            "and do not describe anything outside the window."
        )
    if shot.get("role") == "aroll":
        lines.append(
            f"This is an A-ROLL take (people talking on mic; the transcript is "
            f"handled elsewhere). Describe it as VISUAL coverage: staging, who "
            f"is on screen and their body language, framing, setting — and in "
            f"cut_notes whether stretches of this {unit} could double as b-roll."
        )
    if motion_gloss:
        lines.append(f"Measured motion: {motion_gloss}.")
    else:
        # Without this, the model reports measurements it was never given:
        # "no significant optical flow is detected" appeared on windows of a
        # film that has no motion.json at all (Sept 2026 calibration).
        lines.append(
            "No measured motion statistics are supplied for this unit. Do not "
            "mention optical flow or claim that motion was measured or detected; "
            "describe only movement visible between the ordered frames, and say "
            "when the samples are too far apart to tell."
        )
    lines.append(f"Frames are in temporal order, sampled evenly from start to end of the {unit}.")
    return "\n".join(lines)
