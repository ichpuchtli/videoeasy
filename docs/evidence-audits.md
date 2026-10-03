# Frame evidence and tag consistency audits

The audit is opt-in. It does not replace direct image tagging and it does not
rewrite existing annotations. It keeps each frame's description separately,
then asks a local model to compare the final tag against them for omissions,
contradictions and unsupported claims. It helps a person review tags; it does
not guarantee that the tags or the evidence are true.

## Run one bounded audit

```sh
lms ps
uv run videoeasy check --config config.<film>.yaml --audit
uv run videoeasy audit --config config.<film>.yaml --only <shot_id>
```

The shot must already have extracted, graded frames and an annotation. The
audit works from those cached frames, and originals are read-only. LM Studio
must be running with its native `/api/v1/chat` endpoint and a model that
supports `reasoning: off`. The audit is not part of `videoeasy all`. It never
starts an ingest or re-tags anything.

`check --audit` tests both the normal vision route and the audit's native
route with a synthetic image before any footage is audited.

Run one audit at a time. Three concurrent calibration jobs on one instance
produced context errors, and retrying their checkers one at a time succeeded
with the same saved observations. A failed check is resumable without
`--force`.

Treat a finding as something to check against the actual frame or playback,
not as an automatic edit. Sparse frames cannot verify precise trim ranges,
complete camera paths, a whole take's stability, speech or other sound.

## Stored evidence

```text
work/evidence/<shot_id>/
  ledger.json       timestamped per-frame observations and provenance
  audit.json        consistency findings and status
  attempts/         every model response and error, retained
```

Frame ids and source-relative timestamps make each claim traceable to a
specific input image. Fingerprints of the images, prompt, model and
annotation stop an old audit from being reused after its inputs change. A
timestamp inferred from the legacy sampling algorithm is labelled inferred,
never presented as a recorded measurement. Raw model descriptions stay
unverified even when their JSON validates.

Legacy caches have no authoritative extraction manifest, so their frame count
comes from the files present. Gaps in the numbering, and a drop from an
earlier ledger, are rejected, but a trailing frame missing from the start
cannot be detected. The configured grade is recorded as an assumption about
the cached images, not as proof of how they were graded. The model identifier
is recorded; no checksum of the weights is.

| Status | Meaning |
|---|---|
| `unchecked` | no current successful audit: invalid inputs or output, or a failed or stale check |
| `needs_review` | the checker raised one or more findings |
| `no_issue_detected` | the checker reported no issue; this does **not** mean footage-verified |

Exit codes: `0` no issue detected, `1` unchecked or error, `2` findings need
review. Command-line usage errors can also exit `2`, so read the printed
result as well.

Every finding must cite known frame-evidence ids. An invalid schema or an
invented evidence reference cannot produce a clean status. A failed observer
cannot silently drop a frame and let the remaining frames pass as complete
evidence.

`--force` requests new model work while keeping prior attempt records. After
an annotation edit, audit again, because the old consistency result no longer
applies.

### Transport and budgets

The observer and checker use LM Studio's native API with reasoning and
conversation storage disabled on each request. Global server settings and
model precision are untouched, and there is no silent fallback to another
backend. Both stages default to 1,536 output tokens. They do not inherit the
annotation pipeline's larger budget. Overrides per config:

```yaml
evidence_audit:
  observer_max_tokens: 1536
  auditor_max_tokens: 1536
```

Only integers from 128 to 32,768 are accepted. A cap does not guarantee the
request fits the loaded context, because input tokens count too. Unexpected
reasoning, missing completion statistics, a response that hits its cap,
context errors and malformed responses all stay `unchecked`, with the
response retained. Nothing is silently truncated to make a request fit.
Changing only the auditor budget reuses the frame observations and reruns the
checker. Changing the observer budget invalidates the observations too.

## Bible integration

On the next `assemble`, the bible includes current evidence, review status
and findings beside the original annotation. Frame observations stay
available as individual records instead of being folded into another lossy
summary, and missing or stale audits are marked unchecked.

## What this does not establish

The observer and checker can both make mistakes, including agreeing on the
same wrong description. In calibration, correctly observed empty chairs and
roofs disappeared during synthesis, and dark clothing was misidentified as a
glove. A three-shot calibration produced false positives and missed a roof
omission, although the ledger kept the roof description. Treat the audit as a
record of evidence plus fallible help with review, not as a substitute for
looking at the footage. Start with a few shots, review whether it is useful,
and only then consider broader auditing.
