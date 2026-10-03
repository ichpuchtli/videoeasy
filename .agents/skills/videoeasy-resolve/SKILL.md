---
name: videoeasy-resolve
description: Drive DaVinci Resolve through its Python scripting API or Blackmagic's native MCP from videoeasy — organising media-pool bins, flagging clips for review, laying a cut into a timeline, reading it back and conforming it, audio and grade passes, and rendering. Use whenever code touches src/videoeasy/resolve.py, src/videoeasy/layin.py, the Resolve API or the davinci-resolve MCP, because that API fails silently in ways that look like success. Not for ingest or story work.
---

# videoeasy ↔ DaVinci Resolve

`src/videoeasy/resolve.py` (bins, flags) and `src/videoeasy/layin.py`
(plan / lay / readback / conform / render) are the bridge. The API is usable,
but it lies: under common conditions **writes return None and change nothing
while reads keep working normally**. Assume nothing succeeded until you have
re-read it. The full reference, with evidence, is
`docs/resolve-scripting.md`. This skill is the operating checklist.

Requires Resolve Studio running with external scripting enabled
(Preferences → System → General → External scripting using: Local).

## Before any write

1. **Page gate.** If `resolve.GetCurrentPage()` is `None`, a modal dialog or
   the Project Manager is open, and every write will silently no-op. Stop and
   ask the editor to close it. There is no programmatic way around it.
2. **Project guard.** Use `get_project(expected_name=...)` with the name from
   `film.yaml` `layin.project`. One project may hold several films: an
   unguarded bulk operation can scramble the others.
3. **Not rendering.** Never read a timeline back while
   `IsRenderingInProgress()` is true: property reads return `None` and the
   conform fails with nothing wrong.
4. **Confirm bulk mutations.** Say what will change and how many clips it
   affects, then do it.

## Verify every write by re-reading

- `MoveClips`' return value is unreliable. Re-read the target bin and compare
  names or, better, unique media ids.
- Confirm `AddSubFolder` in `GetSubFolderList()`.
- After timeline writes, re-read the track item lists before the next step.
- `layin` already does this. Scripts sent through the MCP's `run_script` must
  do it themselves.

## Traps that look like success

- **No "Red" clip colour** (16-colour palette). `SetClipColor("Red")` fails
  quietly. Use `clip.AddFlag("Red")` for review marks.
- **Placement:** `AppendToTimeline` with `recordFrame` is the only
  non-rippling primitive. `startFrame`/`endFrame` are zero-based source
  frames at the clip's own rate, with `endFrame` exclusive. `recordFrame`
  includes the start timecode (86400 at 24 fps). `mediaType: 1` places video
  only.
- **Never insert titles on a populated timeline.** `InsertTitleIntoTimeline`
  and `InsertFusionTitleIntoTimeline` ripple every track and marker and split
  clips. Placeholders are rendered title-card clips, imported and appended.
- **One marker per frame.** `AddMarker` returns `False` on a collision.
- **`DeleteClips` leaves linked audio.** Delete the audio item at the same
  start too.
- **`DeleteTimelines` fails on the current timeline.** Switch first.
- **Units:** `GetLeftOffset` reports timeline frames while `AppendToTimeline`
  took source frames (×24/23.976 for drone footage). `GetSourceStartTime()` is
  source-timecode seconds, not media time.
- **Audio:** `AudioVolume` caps at +30 dB, and `NormalizeAudioLevel` stops
  there too. Dialogue Leveler properties are Active Timeline Only (mode 0
  wider, 1 moderate, 2 more lift, 3 whispery). A channel map written through
  the API is **not rendered until the timeline is reopened**: `layin` reopens
  before every render and measures left against right per mono pick.
- **Grade:** `SetCDL` wants space-separated strings (`"1.08 1.08 1.08"`);
  Python lists render solid red. A CDL offset on the LUT node acts before the
  LUT, in log space (an exposure pull). There is no CDL read-back, so
  calibrate on stills from `ExportCurrentFrameAsStill` (works from the Edit
  page) and check with the scorer's exposure measure.
- `SetName` failed on names containing a colon. `SaveProject` lives on the
  project manager.

## Laying a cut (`layin.py`)

```bash
uv run python -m videoeasy.layin --film data/films/<film> --cut editorial/cut-vN.json --plan-only
uv run python -m videoeasy.layin --film data/films/<film> --cut editorial/cut-vN.json \
    --name "<film> cut-vN (AI)" [--replace] [--render <stem>]
uv run python -m videoeasy.layin ... --conform-only | --render-only --render <stem> | --leveler-only
```

- A cut without `dest_in_s`/`dest_out_s` is refused. Rebuild it with
  `cutbuild`. Placement is never re-derived by laying rows consecutively.
- A-roll picks alternate V1+A1 and V2+A2, each extended 6 frames into the
  source and capped at the neighbouring word, with `SetFades` equal to the
  overlap as the crossfade (the API has no transition call). Beat gaps take
  the outgoing take's own tail, and cover goes on V3.
- Conform matches rows to read-back items by name, track and record frame
  within ±2 frames, then compares duration, in-point, fades, gain and channel
  map. Then `intent.check_readback`, which makes `layin` exit 2 on a broken
  rule even when the conform passed.
- Outputs go to `editorial/lay/<cut>.{plan,readback,conform,manifest}.json`.
- Verify joins by measurement (`blackdetect`, `silencedetect` in `cuteval`),
  never by watching.

## Native MCP (`davinci-resolve`)

Blackmagic's MCP bundle (`DaVinciResolve.mcpb`, inside the Resolve app
bundle) is extracted to `~/.claude/mcp-servers/davinci-resolve/` and
registered from `.mcp.example.json` → `.mcp.json`. It is not redistributed
here. Call `get_whats_new` first in a session. `run_script` injects `resolve`
and `project`, and its sandbox does **not** make scripts read-only, so apply
every guard above inside the script. Return metadata and text, never media,
to the frontier model.

## Safety

- Media-pool operations move references and never touch files on disk.
- AI timelines carry an `(AI)` suffix and live in the drafts bin from
  `film.yaml`.
- Source paths are baked into the media pool, so keep drive mount names
  stable.
- Diagnostics go on a copied timeline with a unique review name. Restore the
  active timeline and render preset afterwards, and say what you left behind.
