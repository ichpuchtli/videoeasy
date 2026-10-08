# Face identity

A participant's before/after reel needs that person's own moments: the
interview, the B-roll of them in the process, the testimony. The face tracks
from `videoeasy.faces` are anonymous (`t03` of one clip is unrelated to `t03`
of another), so something has to say which tracks show the same person.

`videoeasy.identity` turns each track's clearest views into face embeddings,
proposes which tracks are one person, and leaves every name to the editor.
It either groups the tracks among themselves or matches them to reference
photos. The editor confirms or drops each proposal on a local page. Only a
confirmed link names a person anywhere downstream.

## Licences

videoeasy is MIT. The recognition model is not part of it.

- `identity setup` downloads InsightFace's `buffalo_l` pack (SCRFD detection,
  ArcFace `w600k_r50` recognition) from InsightFace's own v0.7 release into
  `~/.cache/videoeasy/models/insightface`, and checks it against a pinned
  sha256. The weights are never stored in or shipped with this repository.
- **InsightFace's code is MIT, but "the pretrained models provided with this
  library are for non-commercial research only, whether downloaded
  automatically or manually"** (its python-package README, checked 8 Oct
  2026). That licence binds whoever runs the model. `setup` and `run` print
  it, and every identity manifest records it beside the model's name and
  hash.
- `identityworker.py` runs the model in its own pinned environment (a PEP 723
  script: `insightface` 1.0.1, onnxruntime 1.28, CPU), the same way the face
  cue worker runs MediaPipe. The project's own dependencies never include a
  recognition library. A test checks both, and checks that no weights sit in
  the repository (`tests/test_identity.py`, `Licence`).

For work the non-commercial licence doesn't cover, use a model whose licence
allows it. Run your own tool on the exported request and import its answer
(see [Bring your own model](#bring-your-own-model)). One permissive option is
SFace in the OpenCV model zoo, whose model directory states Apache-2.0. That
page doesn't say what data it was trained on. A commercially licensed model
is the other route.

**Privacy.** Identity reads who people are. The crops, the embeddings
(`work/identity/embeddings.npz`, biometric data), the answer and the
decisions stay under `data/` (git-ignored), are never published, and never
reach a hosted model. A link ties a face track in one clip to a label the
editor typed. Labels are never collected into a profile across projects.
Whether to link people at all is the editor's call for each project. Delete
`work/identity/` to remove every crop and embedding.

## Steps

```bash
uv run python -m videoeasy.identity setup                          # once per machine
uv run python -m videoeasy.faces    --config C                     # face tracks (see register.md)
uv run python -m videoeasy.identity run     --config C [--people DIR] [--same 0.45]
uv run python -m videoeasy.identity review  --config C             # local page: label each proposed person, drop wrong faces
uv run python -m videoeasy.identity confirm --config C --decisions ~/Downloads/identity-decisions-<day>.json --by <editor>
uv run python -m videoeasy.register find    --config C --person <label> [--register joy] [--silent]
```

**`run`** does four things:

1. **Export.** A track qualifies when it has a readable sample of `MIN_PX`
   (80 px, provisional) or more. Its crops are up to three of its most
   frontal and largest readable samples, at least a second apart. They are
   graded like every other frame a model sees, cut at 1.8 face-widths and
   saved at 320 px.
2. **Embed.** The worker embeds each crop that is new, using the face nearest
   the crop's centre. A tight crop is padded and detected again before it
   counts as faceless. Embeddings are cached per crop and model, so a second
   run only embeds new crops.
3. **Propose.**
   - A track's embedding is the normalised mean of its crops'.
   - With `--people DIR` (one folder of reference photos per person, named by
     the person's label, using the largest face in each photo), a track whose
     cosine similarity to a person's centroid is at or over `--same` is
     proposed as that person.
   - The other tracks are grouped by average-linkage clustering on cosine
     distance, cut at `1 - --same`. A group of one track is no proposal.
   - The grouping is code in the project (scipy). The worker only measures.
4. **Import** the answer, with the checks below.

**Review.** The page shows one box per proposed person, with one crop per
track. You type a label into a box to confirm it: every face you keep is
linked to that label, and every face you click to drop is recorded as *not
this person*. Boxes with no label record nothing. A label can be a name or a
code from your own list. People you have already confirmed come first,
under their label. Decisions stay in the browser until you download them.

## What is checked in code

- An answer made for another export (`request_id` differs) is refused.
- A track whose source has had `faces` measured again since the export is
  `unchecked`. Track ids are positions within a clip
  ([lessons.md](lessons.md) 6.5), so an old answer could name the wrong face.
  The same rule covers decisions: each one is bound to the faces measurement
  it was made on, and stops applying when that measurement changes.
- Two tracks proposed as one person that share more than 0.5 s of one clip
  are both flagged (`conflict`), because one face can't be in two places.
  The review page marks them *overlap*.
- Keys the request never had, and malformed entries, are counted and
  reported. A requested track with no answer stays `unchecked`.
- Link statuses: `proposed`, `none` (nothing matched it), `confirmed`,
  `rejected`, `unchecked`, `invalid`. Only `confirmed` carries a person.

## Bring your own model

`identity export` writes the request alone. Any tool that reads it and
writes an answer can stand in for the built-in model, and `identity import`
then reads that answer with the same checks.

`work/identity/request.json`:

```json
{"schema": "videoeasy.identity.request/1", "request_id": "3b9786032d82f3cb",
 "crop_px": 320, "min_px": 80,
 "sources": {"<source id>": "<faces evidence file>"},
 "tracks": [{"key": "<source id>/t02", "source": "<source id>", "track": "t02", "t0": 31.2, "t1": 44.0, "px": 185,
             "crops": [{"file": "crops/<source id>/t02-<hash>.jpg", "t": 39.6, "box": [812, 240, 186, 186]}]}]}
```

The answer:

```json
{"schema": "videoeasy.identity.answer/1", "request_id": "3b9786032d82f3cb",
 "tool": {"name": "...", "model": "...", "model_licence": "..."},
 "people": {"g001": {"label": null}},
 "tracks": [{"key": "<source id>/t02", "person": "g001", "score": 0.71},
            {"key": "<source id>/t05", "person": null, "why": "no other track matched"}]}
```

- `request_id` is copied from the request.
- `person` is the tool's own id for a person, or null when it can't say.
  `score` is optional.
- `tool.model_licence` goes into the import manifest.

`out/identity.json` holds the imported proposals. It can be rebuilt from the
answer at any time. `<film>/editorial/identity-decisions.json` holds the
editor's decisions: who decided, when, and what each one superseded. It is
human judgment, like `intent.json`, and is never regenerated.

## First test (8 Oct 2026)

- **Setup.** It downloaded the 288 MB pack in about a minute, matched the
  pinned hash, built the worker environment and loaded the model.
- **InsightFace's own bundled sample photos.**
  - Six faces were cut into two "clips", one flipped and scaled. All six
    pairs grouped correctly at `--same 0.45`.
  - A reference folder made from one crop matched both of that person's
    tracks.
  - A tight 112 px reference face was found once padded, and matched nobody,
    which is correct.
- **The whole chain on video.**
  - Two 4 s clips were made from the same group photo, one mirrored, and run
    through `faces` and `identity run`.
  - 10 tracks qualified, and their 30 crops embedded in 2 s on the CPU.
  - All five people present in both clips were proposed as five correct
    pairs, with the mirrored order resolved. A second run embedded nothing
    new.
- **Export, on one event clip.** Two of its four tracks had a frontal,
  readable view of 80 px or more. Expect many observational B-roll tracks to
  have no recognisable view: most group faces measured 30–60 px
  ([register.md](register.md)).
- Recognition hasn't been run on any event's participants yet. `MIN_PX` and
  `--same` are provisional until it has been, and until the editor's
  confirmations show how often a proposal was wrong.

## Not built yet

- Tracks nothing matched (`none`) are not on the review page, so a person
  seen in only one track can't be named there.
- Placement by person (a participant's reel from their own confirmed
  moments) is part of register-aware placement ([register.md](register.md),
  Placement).
