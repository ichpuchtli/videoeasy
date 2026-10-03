# DaVinci Resolve scripting: what holds and what lies

Everything learned driving DaVinci Resolve Studio 21.1 from Python, both
through the external scripting API (`src/videoeasy/resolve.py`,
`src/videoeasy/layin.py`) and through Blackmagic's native MCP server. Every
item was observed on a live project, and most were found because a script
reported success while nothing changed.

**The single rule:** assume no write succeeded until you have re-read it.

Contents: [Setup](#setup) · [Silent write failures](#silent-write-failures) ·
[Media pool](#media-pool) · [Placing clips on a timeline](#placing-clips-on-a-timeline) ·
[Joins](#joins) · [Audio](#audio) · [Grade](#grade) ·
[Read-back and conform](#read-back-and-conform) · [Rendering](#rendering) ·
[Native MCP](#native-mcp) · [Safety](#safety)

---

## Setup

- Resolve must be running, with Preferences → System → General → External
  scripting using: **Local**.
- The scripting module and the API stubs live under
  `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/`
  (`DaVinciResolveScript.pyi` is the most reliable reference to what exists).
- Studio is needed for external scripting.

## Silent write failures

- **If the Project Manager or any modal dialog is open, `GetCurrentPage()`
  returns `None`, and every write silently returns `None` and changes
  nothing**: `AddSubFolder`, `MoveClips`, `OpenPage`, timeline edits. Reads
  keep returning correct data, so the script looks healthy right up to the
  point where you notice nothing happened. There is no programmatic way
  around it: ask the person at the machine to close the dialog.
- Gate every write session on it:

  ```python
  if resolve.GetCurrentPage() is None:
      raise RuntimeError("Resolve is on no page: a modal dialog or the "
                         "Project Manager is open. Close it and retry.")
  ```

- **While Resolve renders, item property reads return `None`**
  (`GetProperty` and `GetProperties` alike), and `GetCurrentPage()` is also
  `None`. A read-back taken during a render reports every gain and leveler
  as missing and fails a conform that has nothing wrong with it. Never read a
  timeline back while `IsRenderingInProgress()` is true. Render only after
  the read-back and conform have passed.

## Media pool

- **`MoveClips`' return value is unreliable.** A truthy return does not mean
  the clips moved; a synthetic silent-failure pool returned `True` with the
  destination still empty. Re-read the target bin and compare:

  ```python
  before = {c.GetName() for c in target.GetClipList()}
  media_pool.MoveClips(clips, target)
  after = {c.GetName() for c in target.GetClipList()}
  assert after - before == {c.GetName() for c in clips}
  ```

  Prefer unique media ids over display names when matching.
- Confirm `AddSubFolder` by finding the folder in `GetSubFolderList()`
  instead of trusting its return.
- **The clip colour palette has 16 entries and "Red" is not one of them.**
  `SetClipColor("Red")` fails, in the same quiet way, even when every other
  write works. Use flags for review marks: `clip.AddFlag("Red")`.
  `SetClipColor("Orange")` and `("Pink")` work on timeline items.
- Media-pool operations move references inside the project and never touch
  files on disk, which makes bin reorganisation safe to retry.
- Bin membership is not proof of which film a clip belongs to when one
  project holds several films that share camera bins.
- `SaveProject` lives on `resolve.GetProjectManager()`, not on the project.

## Placing clips on a timeline

- **`MediaPool.AppendToTimeline` with `recordFrame` is the only placement
  primitive that does not ripple.** Pass a list of clip-info dicts:
  - `startFrame` / `endFrame`: zero-based **source** frames at the clip's own
    rate (24, 23.976, ...), with `endFrame` exclusive (duration = end −
    start).
  - `recordFrame`: an absolute timeline frame *including* the timeline start
    timecode (01:00:00:00 at 24 fps is 86400).
  - `trackIndex` picks the track. `mediaType: 1` places video only, which
    keeps a B-roll clip's scratch audio off the audio tracks. A full-clip
    append brings linked audio onto A1.
  - One list for a whole track is fast, and the read-back matches frame for
    frame.
- **Never call `InsertTitleIntoTimeline` or `InsertFusionTitleIntoTimeline`
  on a populated timeline.** They are insert edits at the playhead: they
  ripple every track and every marker by the title's length, split the
  clips under the playhead on the other tracks, and land on whichever track
  Resolve currently treats as the destination (V1 once, V2 the next time).
  **Placeholders are rendered clips instead:** a title card made with Pillow
  and ffmpeg (12 s, 1920×1080, 24 fps), imported with `ImportMedia` into a
  drafts bin and appended like any other clip, so it fills its gap at the
  exact hold.
- **One marker per frame.** `AddMarker` returns `False` when a marker already
  sits on that frame. Offset by a frame and re-read `GetMarkers()`.
- `TimelineItem.SetName` failed on names containing a colon and succeeded on
  the same names with " - " (observed, not root-caused).
- `Timeline.DeleteClips([video_item])` does **not** remove the linked audio.
  Find the audio item at the same start, delete it too, and re-read both
  tracks.
- `MediaPool.DeleteTimelines([tl])` returns `False` for the current timeline.
  Make another timeline current first, then confirm the name has gone from
  `GetTimelineByIndex` before recreating it.
- An order of operations that survived: create the timeline with the drafts
  bin current → append the A-roll picks → beat markers → cover on a higher
  track from each row's destination → placeholder clips in their gaps →
  markers → save. Re-read each step's track item list before the next step.

## Joins

Butting picks on one track with a short fade on each is a dip to zero at
every cut. Room tone left as an empty gap is black and silence.

- **Alternating tracks.** Consecutive A-roll picks go on V1+A1, then V2+A2.
  Each pick is extended 6 frames past its padded in and out, so neighbours
  overlap by 12 frames, and `SetFades({"FadeIn": n, "FadeOut": n})` equal to
  the overlap makes a real crossfade. The API has no transition call.
- **Cap each extension at the nearest neighbouring word** in the source
  transcript, because a stray syllable is worse than a shorter fade. When the
  outgoing tail is capped, the incoming head grows instead.
- A pick cut inside a phrase carries its own measured room (from source
  energy per channel). A pick with no head room starts on its word with a
  2-frame fade-in, and the outgoing tail fades out under it.
- **Beat gaps are filled with the outgoing take's own tail** (room tone from
  the same recording). The beat's last cover clip is extended across the gap
  so the picture never falls to the lower track for 12 frames. Any residual
  word-capped hole gets a video-only slice of the outgoing take, planned only
  when no cover already spans the hole; otherwise Resolve drops the slice as
  an overlap.
- **Match items back to the plan within ±2 frames.** 23.976 fps rows push
  later items by a frame, and an exact match once silently skipped 11 items,
  which then rendered unbalanced.
- **Measure, don't watch:** `blackdetect` and `silencedetect` on the render.
  This layout took one render from 30 black stretches and 24 dropouts to 1
  and 3. A later one-frame black came from rounding start and length to
  frames separately, so plan every join gap in placed frames.

## Audio

- **Fades:** `item.SetFades({"FadeIn": 5, "FadeOut": 5})` on audio items, in
  frames. Read back with `GetFades()`.
- **One-sided sources:** find them first with ffmpeg per-channel RMS (a right
  channel at −84 dBFS is dead). Then
  `item.SetSourceAudioChannelMapping(json)`, keeping the rest of the JSON from
  `GetSourceAudioChannelMapping()` and setting
  `track_mapping["1"] = {"channel_idx": [1], "mute": false, "type": "mono"}`.
  A mono clip on a stereo track plays centred, eventually (next item).
- **A channel map written through the API is stored but not rendered until
  the timeline is reopened.** The map read back correctly on every mono item,
  and two full renders still put those picks on the left only (the right
  channel 12 to 30 LU down). The signature is integrated level 3 LU low with
  true peak unchanged. Switching to another timeline and back before
  rendering fixed it. Measure left against right per mono pick on the
  rendered file; the read-back cannot catch this.
- **Clip gain caps at +30 dB.** `SetProperties({"AudioVolume": dB})` accepts
  larger values and silently reads back as 30. `Timeline.NormalizeAudioLevel`
  stops at the same 30: it returns `True` and leaves the clip at 30. Quiet
  camera scratch audio (−50 to −62 LUFS raw) can need 31 to 38 dB to reach
  −24 LUFS. More gain is not available, so the fix is a decision about target
  and dynamics.
- **Peak control is clip gain only.** The API gives no track, bus or limiter
  access. With camera audio at crest factors over 20 dB, −18 LUFS clipped in
  113 places; the peak-safe, balanced level was about −24 LUFS. Gains per
  pick: `min(30, −24 − LUFS, −1.5 − peak)` from ffmpeg `ebur128` on the
  source range, using the left channel when the right is more than 8 dB down.
  Render, measure per pick, correct once.
- **Dialogue Leveler** (`AudioDialogueLeveler*` properties) is **Active
  Timeline Only**: make the timeline current first. `GetProperties` on a
  non-current timeline's items returns only `AudioVolume`. Mode constants
  (21.1): allow wider dynamics 0, optimize moderate levels 1, more lift for
  low levels 2, lift soft whispery sources 3. It adds up to 6 dB of output
  gain. Measured on diagnostic renders:
  - **moderate (1):** 8.9 LU spread across picks, peaks at −1 dBTP;
  - **allow wider (0):** 10.9 LU spread;
  - **the two lifting modes (2, 3):** lifted already-loud picks to −16 LUFS
    with +3 to +5 dBTP clipping, and barely moved the quiet ones.

  In an earlier test the leveler properties read back, but changing them had
  no measurable effect on renders (output gain 6 → 0 left loudness
  identical). The measurements above came later, with the timeline current.
  "Active Timeline Only" is the likely explanation, but it is not confirmed.
  Verify any leveler change on a render.
- Item order on A1 equals pick order, so per-pick lists can be zipped onto
  `GetItemListInTrack("audio", 1)`. Match items across timelines by
  `(GetName(), GetSourceStartFrame())`.

## Grade

- `item.GetNodeGraph().SetLUT(1, "/abs/path.cube")` accepts an absolute path
  outside Resolve's LUT folder and reads back as a relative one. Verify with
  `GetLUT(1)`.
- **`item.SetCDL({...})` wants space-separated strings** for Slope, Offset and
  Power (`"1.08 1.08 1.08"`). Python lists are accepted without error and
  render as a solid red frame.
- **A CDL offset on the LUT's node acts before the LUT, in log space.** It is
  an exposure pull: −0.05 moved the shadows 22 code values and the highlights
  4. Per-shot exposure trims are applied this way.
- **There is no CDL read-back.** Calibrate trims on Resolve's own graded
  stills: `ExportCurrentFrameAsStill` works from the Edit page. Then check
  them with the scorer's exposure measure on the render.
- **`GetSourceStartTime()` answers in source-timecode seconds, not media
  time.** Take the span an item shows from its plan row instead.
- Check a grade change on a 30 s `MarkIn`/`MarkOut` range render (a few
  seconds) before rendering the whole timeline.

## Read-back and conform

`layin.py` has three parts: `plan` (frame-exact placement from the cut's
`dest_in_s`/`dest_out_s`, no Resolve), `lay` (the writes, each verified by
re-reading), and `readback` + `conform`. Conform matches every planned row to
a read-back item by name, track and record frame within two frames, then
compares duration, source in-point, fades, gain and channel map, and lists
unmatched rows and unexpected items. Nothing is inferred from a total
duration: a correct duration does not detect swapped clips or early cover.

- **Units differ between write and read.** `AppendToTimeline` takes the
  in-point in source frames, while `GetLeftOffset` reports it in timeline
  frames. A drone in-point planned at source frame 3842 reads back as 3845
  (× 24/23.976). Convert before comparing, or every drone row looks three
  frames off.
- A cut without destination intervals is refused. The lay-in never re-derives
  placement by laying rows consecutively.
- After conform, the read-back is checked against the editorial rules
  (`intent.check_readback`). The A-roll's own picture sits beside its sound
  and is not cover. `layin` exits 2 when a rule is broken on the timeline,
  even when the conform passed.
- A live round trip on a scratch timeline with three frame rates, mono and
  stereo sources and a placeholder card conformed 33 of 33 items.
- A saved lay manifest listed 92 cover rows while the live read-back held 89
  (three short fillers dropped as overlaps). Store the read-back used for each
  render, and never treat a manifest as proof of placement.

## Rendering

- `render_timeline` reopens the timeline first (see channel maps), renders
  with the film's preset, waits for the job, then measures left against right
  per mono pick on the file.
- Range renders for diagnostics: `SetRenderSettings` with
  `SelectAllFrames: False` and `MarkIn`/`MarkOut` in timeline frames including
  the start timecode. A 60 s range renders in a few seconds. An 18-minute
  1080p H.264 timeline took about ten minutes.
- Leave diagnostics on a copied timeline with a unique review name, restore
  the active timeline and render preset afterwards, and record that you did.

## Native MCP

DaVinci Resolve Studio 21.1 ships its own MCP server:

- the bundle is at
  `/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Resources/DaVinciResolve.mcpb`,
  and the native executable is under `.../Contents/Applications/ResolveMCP`;
- it declares 14 tools: API search, API stubs, developer docs, what's new,
  status, launch, sandboxed Python execution (`run_script`, and
  `run_script_unsafe`), and LUT and DCTL management. Its Node wrapper proxies
  to the native executable;
- `run_script` runs bundled Python 3.14 with `resolve` and `project`
  injected. Its OS sandbox does **not** make scripts read-only: Resolve API
  mutations still work. Apply the same guards as `resolve.py` (page gate,
  project-name guard, read-back verification).

Setup used here: extract the `.mcpb` bundle (a zip) to
`~/.claude/mcp-servers/davinci-resolve/` and register it with Claude Code. A
template is in [`.mcp.example.json`](../.mcp.example.json):

```json
{"mcpServers": {"davinci-resolve": {"type": "stdio", "command": "node",
  "args": ["${HOME}/.claude/mcp-servers/davinci-resolve/server/index.js"]}}}
```

Copy it to `.mcp.json` (which is git-ignored). **The bundle is Blackmagic's
software and is not redistributed in this repository:** extract it from your
own Resolve installation.

Parity check: one timeline snapshot (item names, timeline and source
start/end, source paths across two video tracks and one audio track) read
through the MCP and through the external Python bridge produced identical
normalised JSON, about 165 ms each. That establishes working metadata access.
It does not establish mutation reliability, conform accuracy for retimes or
nested clips, or better editorial decisions. Call `get_whats_new` early in a
session: Resolve adds API features faster than any model's training data.

Keep the local-media boundary when using the MCP: return metadata and text to
the frontier model, and route any image or audio analysis to local models.

## Safety

- **Guard the project name** on every mutating session
  (`get_project(expected_name=...)` refuses the wrong project). One project
  may hold several films, so an unguarded bulk operation can scramble the
  others.
- **Confirm before bulk mutation.** Say what will change and how many clips
  it affects, then do it.
- AI timelines carry an `(AI)` suffix and live in a drafts bin. An in-place
  patch that keeps every duration the same leaves the rest of the timeline
  and its markers untouched.
- Source paths are baked into artifacts and media-pool references, so keep
  drive mount names stable.
