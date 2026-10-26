# Plan A MCPO private executor — amended execution brief

Authority: direct DC execution approval, 2026-10-26, exact Amended Packet.
Approval reference: `DC-2026-10-26-AMENDED-PACKET`. This reference is a binding
identifier, not a credential, new approval mechanism, or replacement for DC consent.

## Scope, owners, G0 and clock

Full DC Definition of Done: deliver a usable private executor and complete the
approved one real cohort replacement with protected changed-edge `/_tools`
acceptance. The frozen eight-case Red/Green group and removal of MCPO startup
key-fragment logging are required local implementation evidence, not substitutes
for that full outcome. **FULL OUTCOME: NOT MET.** Local Green does not establish
runtime readiness, real replacement or protected acceptance.

Maximum: exactly four authored files, no generated caches or additional artifacts:

1. `docs/Plan_A_MCPO_Executor_Brief_2026-10-25.md`
2. `tools/plan_a_executor.py`
3. `tests/test_plan_a_executor.py`
4. `src/mcpo/main.py` — remove startup key-fragment logging only; preserve auth,
   routes, configuration, versions, startup behavior and presence-only logging.

Active assurance: **accidental-drift assurance**. Concrete scenario: the trusted
operator accidentally targets the wrong resource or changes unrelated settings
while performing the approved cohort replacement. Logging and retry tests are
safeguards, not additional or expanded G0 control-3 scenarios. No
adversarial-contributor or enforcement-core expansion is selected.

One bounded correction cycle: author and freeze eight cases; run Red; apply one
bundled executor/main/brief correction; run the full group once for Green. Maximum
two full local suite invocations total. A failed Green means STOP, not another fix.
No post-Green software/test edit, refactor, confirmatory suite run, review-agent
launch or automatic second material correction is permitted. Direct DC separately
authorized this non-material brief-only closeout accuracy amendment: truthful
recording of the unchanged candidate does not begin material correction cycle 2.
AST-only syntax/source inspection is not a suite; none is rerun for this amendment.

L2 window: `2026-10-26T15:18:24Z` to checkpoint `2026-10-26T16:03:24Z` (45 minutes).
Aim code/Green by `15:43Z`; no autonomous extension past checkpoint. No remote build,
CI rerun, installation, model/provider change or live API/credential call is in this
authoring run. Existing Python/stdlib tooling only, always `python -B`.

SA-04 is sole writer of these four files. Root L2 owns coordination, qualifications
and the private operator boundary. SA-02 owns Railway runtime evidence, secret
write/activation and deployment qualification. SA-03 owns Git publication and
publisher pause/restore. No independent review was requested. No commits, staging,
push, PR, merge, workflow change, live key generation, DPAPI real-successor test,
production mutation or cleanup occurs in this task.

## Exact targets and provenance (V0)

| Binding | Exact value |
| --- | --- |
| Railway project | `62cb5067-2ad6-47b2-97d6-de206f37d18a` |
| Environment | `565c4c1e-0616-47d2-8fe3-faa6837fbe65` |
| MCPO service | `66391175-bd78-456f-93d5-641498dabcf9` |
| MCPO origin | `https://open-webui-mcpo-production.up.railway.app` |
| Open WebUI origin | `https://open-webui-production-b978.up.railway.app` |
| Open WebUI service, SA-02 safe evidence | `8f7d6379-4efb-42a1-809c-c0e961cb0fdb` |
| MCPO last observed deployment | `c527e170-5719-4d84-96c4-b747655791ab`, SUCCESS |
| WebUI last observed deployment | `ccd3bab6-0df0-4572-a7a8-b7a218303e4e`, SUCCESS |

SA-03 supplied and verified local source/worktree lineage: repository
`C:\Projects\open-webui-mcpo`, exact branch `plan-a/mcpo-executor-2026-10-25`,
clean preimage HEAD/origin-main `e0c1eda4e2fd6446a9fd44f071fd201dd818c4bf`, tree
`49b345cc613846d140578125807e3e0ef468766a`. Feature lineage
`73652333be48a7aebf3ffc12bc58bf7057bbd48b` has the identical tree. Neither identity
proves live deployment source. No reset/stash, overwrite of another writer, or
source cleanup is authorized. Repository is **PUBLIC**; default `main` currently
has no active protection. Do not repeat dated private/protected claims.

MCPO source is `main.py` version `0.0.75` at the exact preimage above. The only
application delta removes the `_mask` block with first/last-four-character key
slices. Auth still compares the supplied bearer key to `API_KEY`; presence-only
`Provided`/`Not provided` remains. The reviewed patched commit is not yet assigned:
Git publication and its actual deployed-source binding remain pending.

WebUI supported consumption is its existing authenticated administrative API:
`GET /api/v1/configs/tool_servers`, preserve/modify the object privately, `POST`
the same route, then private `GET` readback. No browser automation, token store,
login, import/export config endpoint, or generalized admin API is selected.

Official baseline: Open WebUI **v0.8.10**, commit
`e4e69a10ec08a725bf2ab3db499ef664f2bd7570` (root's retained exact-source
qualification), especially
[`backend/open_webui/routers/configs.py`](https://github.com/open-webui/open-webui/blob/v0.8.10/backend/open_webui/routers/configs.py),
[`config.py`](https://github.com/open-webui/open-webui/blob/v0.8.10/backend/open_webui/config.py),
[`env.py`](https://github.com/open-webui/open-webui/blob/v0.8.10/backend/open_webui/env.py),
and [`utils/tools.py`](https://github.com/open-webui/open-webui/blob/v0.8.10/backend/open_webui/utils/tools.py).
The v0.8.10 router source was independently read during authoring. Official API
documentation: <https://docs.openwebui.com/getting-started/api-endpoints/>.
The newer current-main router is NOT substituted for this baseline. Current live
WebUI version/source has not been proved by the available safe runtime evidence;
root/SA-02 must bind it before any private GET. Exact upstream test execution and
live schema/preservation are not claimed; frozen tests target the local algorithm.

v0.8.10 schema: outer `TOOL_SERVER_CONNECTIONS` list; each connection requires
`url`, `path`, nullable `auth_type`, `key`, `config`; `type` defaults to `openapi`
and `headers` defaults to `None`. Entry `extra=allow` preserves unknown/nested
fields. The live adapter rejects absent required fields or unknown outer envelope
fields rather than silently dropping/normalizing them. Comparison only permits
documented missing `type=openapi` and absent/None/empty-dict headers equivalence.
No ordering changes, unknown-field drops, auth coercion, or nested normalization.
This version/schema/backend/surface/scope baseline must be refreshed if changed.

Native Railway baseline supplied by SA-02: installed CLI **4.40.0**;
<https://docs.railway.com/cli/variable> and
<https://docs.railway.com/deployments/deployment-actions>, with exact-tag CLI
`src/commands/variable.rs`, `service.rs` and `redeploy.rs` source qualification.
Existing native authentication remains opaque. No Railway resource, API, auth
store, variable/config export or log is read by this implementation session.

## Private engine contract and ordering

Closed typed request fields only: `operation_id`, `approval_reference`,
`expected_target`, `expected_source` (reviewed 40-hex commit),
`expected_deployment` (current deployment UUID), approved `selectors` tuples
`(url,path,info_id)`, `phase` (`execute` or acknowledged `resume`), and finite
prerequisite confirmations plus their nonsecret qualification reference.
No key/JWT/password input, arbitrary command, destination, path-to-secret or
general HTTP operation can be supplied through CLI/model arguments.

Order in the private engine:

1. Validate exact target/approval/closed schema and prerequisite confirmations.
2. Require a qualified activation owner **before secret mutation**. The shipped
   CLI deliberately fails this preflight before even prompting for JWT or doing
   a GET: `ACTIVATION_OWNER_REQUIRED`.
3. Under private custody lock, handle prior operation state. Reuse the same
   successor for acknowledged READY resume. COMPLETE resume is acknowledgement,
   not a fresh runtime PASS. Every intermediate/intent/uncertain effect stops
   for reconciliation; never regenerate, replay, or restore a stale preimage.
4. Privately authenticated GET; validate full server-wide approved cohort. Each
   selector must match exactly once; duplicates/missing/ambiguous selectors fail.
   Each target must be OpenAPI and bearer-only with a nonempty raw key, no
   `Bearer` prefix/whitespace and no case-insensitive Authorization override.
   Headers with uncertain type fail. All cohort keys must agree. Reject every
   extra same-MCPO-origin or shared-key entry outside the approved cohort.
5. Qualify private custody with a fixed **fake** readiness sentinel. Only after
   authenticated GET, selection/auth checks and readiness may `secrets.token_hex(32)`
   generate exactly one real 256-bit successor. This was NOT done in this task.
6. Atomically persist protected successor + binding/phase metadata; write-ahead
   intent before each external effect. Native fixed-vector secretstdin write is
   `railway variable set API_KEY --stdin --skip-deploys --service <fixed-UUID>
    --environment <fixed-UUID> --json`, `shell=False`, private stdout/stderr capture.
    No CLI secret argument, token export, auth-store/config/variable read or deploy.
   Actual adapter limitation: it checks the child exit code only and does not
   interpret/validate a response acknowledgement. Lost or unexpected acknowledgement
   handling at the actual integration boundary remains unqualified.
7. Qualified owner activates once; require allowlisted source/deployment proof:
   exact project/env/service/reviewed commit, new deployment UUID and SUCCESS.
8. Re-GET and compare the **full exact private preimage immediately before POST**
   under operator exclusive-write window. This is not atomic server-side CAS.
   Preserve every entry, array order, unknown/nested field; change selected keys
   only. POST once, GET privately, compare permitted defaults, revalidate cohort.
9. Return only allowlisted `status`, sanitized `code`, and finite `phase`. No raw
   response, error body, traceback, key, JWT, predecessor, fingerprint, child
    stdout/stderr or arbitrary exception message enters public output.

Actual completion limitation: `execute()` currently reaches its internal COMPLETE
after configuration readback/cohort validation. The observed Green exercises that
path with fake adapters only. The code has **no protected `/_tools` acceptance
probe**, and its live CLI/activation bridge is unconnected. Internal COMPLETE is
not full live DC Done; no real cohort replacement or protected acceptance occurred.

`urllib`/`ssl` use a fixed HTTPS origin, validating default TLS context, no proxy
discovery, no redirects/credential forwarding, 20-second socket timeout and
1-MiB response/request limit; duplicate JSON object keys/invalid constants fail.
There is no automatic network retry.

## Windows custody and private bootstrap

Custody is existing-account **DPAPI CurrentUser** through stdlib `ctypes`:
`%LOCALAPPDATA%\DC\PlanA\MCPO\<operation_id>.dpapi`, protected blob at most
64 KiB. Directory must already exist and be qualified; this program does not
create directories or change permissions. Check current-account ownership,
private allow trustees (current user, SYSTEM, Administrators only), deny/unknown
ACE handling, and no reparse points in the full ancestor chain before creating
files. Recheck file ownership/DACL and reparse safety. Existing broad or unknown
ACL, unsupported platform/API, missing directory, pending file, or failed DPAPI
roundtrip fails closed. No fake platform fallback.

Use a retained one-byte lock file with nonblocking Windows file lock. Protect
then exclusive-create a pending file, flush/fsync, atomically replace protected
state, read under lock. No deletion/cleanup on failure; retain uncertain pending
state for human disposition. Only successor and operation/phase/binding plus
nonsecret activation metadata persist. No JWT, predecessor, full config preimage
or raw subprocess output is saved. Same-machine/account limited; no zeroization,
portable recovery, tested restore, or durable directory-fsync claim is made.

Root-supplied host facts at closeout: an independent native **.NET DPAPI CurrentUser**
synthetic in-memory roundtrip returned `True`, with no persistent state. That is
not validation of the Python `ctypes` custody adapter. The required directory
`%LOCALAPPDATA%\DC\PlanA\MCPO` **does not exist**; the code requires it to exist
and cannot create it. The inspected parent owner is the current user, but it has
an inherited extra unrelated-SID `ReadAndExecute` ACE. No ACL modification or
directory creation was performed. Actual private custody remains **UNQUALIFIED**;
no successor was generated and no real-successor DPAPI state was written.

Private bootstrap building blocks, not a usable current workflow:

1. Root/SA-02 first records actual nonsecret source, logging, trigger, custody and
   auth availability qualification. Flags are assertions, **not evidence alone**.
2. The implemented `private_jwt()` function requires a private local Windows
   terminal and uses a no-echo `getpass` prompt. A JWT must never enter model/tool
   args, environment exports or files. Redirected/non-TTY streams, unsupported
   host or getpass fallback fail closed. The shipped CLI **does not call this
   function and cannot prompt for JWT**, regardless of confirmation flags.
3. `private_jwt()`, `PrivateHTTP(jwt)` and `WindowsCustody(operation_id)` are
   implemented building blocks but **NOT connected to a live engine entrypoint**.
   Connecting and qualifying the owner-mediated activation path remains missing
   implementation, not an operator command to run now. Do not combine fake
   activation objects with live adapters.

Existing fail-closed CLI grammar only, not a bootstrap/run instruction. All values
are nonsecret; `$ReviewedSourceSha`,
`$ApprovedPath`, `$ApprovedInfoId`, `$QualificationReference` must come from
the owner-approved evidence, not invented examples. Repeat `--selector` for the
complete cohort. This syntax cannot enable live execution or a JWT prompt:

```powershell
python -B tools/plan_a_executor.py rotate-mcpo --operation-id plan-a-20261026-001 --approval-reference DC-2026-10-26-AMENDED-PACKET --project 62cb5067-2ad6-47b2-97d6-de206f37d18a --environment 565c4c1e-0616-47d2-8fe3-faa6837fbe65 --service 66391175-bd78-456f-93d5-641498dabcf9 --expected-source $ReviewedSourceSha --expected-deployment c527e170-5719-4d84-96c4-b747655791ab --selector https://open-webui-mcpo-production.up.railway.app $ApprovedPath $ApprovedInfoId --phase execute --qualification-reference $QualificationReference --logging-qualified --patched-source-qualified --exclusive-writer --auth-available
```

**Current expected result:** sanitized `REJECTED / ACTIVATION_OWNER_REQUIRED`,
exit 2, no JWT prompt/GET/DPAPI/real key/file mutation. Actual selector tuples and
future patched deployment UUID remain pending. This is not an operational
rotation command, and passing flags cannot unlock the absent bridge.

## Logging prerequisite and genuine runtime blockers

Before even private GET, root/SA-02 must qualify source-backed logging behavior
for the actual WebUI deployment and workers/sinks: Authorization and tool_servers
body capture exclusions, Redis refresh path activity, effective existing value
filter if needed, and preservation of **existing** audit policy and delivery.
Do not require enabling audit if intentionally off; do not require a new filter
if Redis value-refresh is inactive. `open_webui.env` can normally emit changed
Redis-refresh values at INFO. Conditional supported WARNING is approved only
when SA-02 proves necessity from source/runtime facts and existing audit delivery
is preserved. No default logging mutation, invented per-logger knob, CRITICAL,
AUDIT_NONE, or automatic restoration to INFO. Malformed Redis raw-data ERROR is
a separate residual risk requiring control or explicit bounded acceptance.

Live source/trigger/log-control/project-workspace mapping proof is currently
incomplete; private JWT unavailable. CLI status/auth success is not that proof.
No credential-bearing GET was attempted. Actual private directory custody and
Python `ctypes` DPAPI operation remain unqualified, despite the independent native
.NET synthetic in-memory success recorded above. Native secretstdin delivery and
its interpreted acknowledgement are unqualified. Local fake success cannot
establish any of these facts.

Native exact-source activation is the **genuine integration blocker**. Existing
successful path selected by DC is GitHub-source **Deploy Latest Commit**. CLI
`redeploy` selects an existing deployment; it is NOT an exact-commit selector and
does not prove new source. No fabricated adapter, local upload/archive,
`--no-gitignore`, registry migration, image swap or restart-old workaround exists.
Root/SA-02 must qualify the existing native UI/owner action and source receipt
before connecting an execution bridge. This task stops before secret mutation,
as explicitly allowed by the Amended Packet. The missing live CLI connection is
an **implementation limitation**, not just a missing external confirmation flag:
the full usable private rotation workflow is **NOT COMPLETE / NOT RUNTIME-READY**.
Do not publish this candidate as a completed operational executor on fake Green.

## Source publication, runtime validation and rollback boundaries

SA-03 later publishes **one code event only** after Green and essential
preconditions. Pause and restore publisher workflows to their recorded original
`active` state: `publish.yaml` ID `182507354`, `build-release.yml` ID `182507351`.
No workflow file, branch-protection, source cleanup or unrelated publisher change
is authorized. Existing Docker workflow side effects must not be misrepresented
as a new deployment method; no registry change is selected here.

One-time code-patch rollout keeps the current key unchanged: reviewed source S
reaches approved fork default `main` and remains frozen; use exactly one existing
GitHub-source Deploy Latest Commit trigger. If existing auto-deploy is proved
active and publication triggers that rollout, **do not also trigger manually**.
Routine rotation later stages API_KEY with `--skip-deploys`, then exactly one
Deploy Latest Commit of that same reviewed S. Source publication, image build,
activation, application readiness and end-to-end tool success are distinct facts.

SA-02/root acceptance remains exactly the DC-approved changed edge: after the
approved one real cohort replacement, validate the protected MCPO `/_tools` path
using the successor and retain only sanitized acceptance evidence. No
predecessor-negative test, old-secret test, current backend-tool requalification
or dated tool-count gate is added. Existing **C1 positive PASS / old-key negative
UNTESTED — closed** remains untouched, not reopened or upgraded by local tests.
Plan B containment remains available under its existing separate authority;
aFPM remains paused. The required protected `/_tools` probe is absent from the
current code and has not run.

No automatic rollback. Unknown write/activation/POST effect stops without replay
or stale-config restore. A later rollback requires exact DC approval and private
fresh-state reconciliation; predecessor/full preimage is deliberately not
retained by custody. No guaranteed native Railway rollback is claimed. This local
source change needs no reload/restart now; patched runtime behavior requires the
separately qualified single deployment. No permissions, identity, vault, model,
provider, other credential, n8n runtime, customer work or production action here.

## Frozen acceptance and honest execution evidence

Eight cases (subcases within the fixed group are allowed):

1. Wrong target/approval/extra request fields reject before any writes.
2. Auth failure, ambiguous/missing/duplicate/out-of-cohort/auth-overriding selectors
   and failed prerequisites reject before secret mutation/activation.
3. Non-target entries/order/unknown/nested config preserved; only documented
   defaults equivalent; concurrent preimage change blocks POST.
4. Fake secret sentinels absent from public receipts, stdout/stderr, errors and
   subprocess-boundary exceptions; allowlisted receipt shape only.
5. Custody failure blocks real generation, mutation and activation.
6. Acknowledged completed resume reuses persisted same successor, with no new
   key generation or replay and no stored predecessor/JWT.
7. Unknown secret-write/POST effects stop; resume does not replay/rollback.
8. AST of actual application source (no app/network imports) rejects API_KEY
   slicing and preserves presence-only logging.

Actual Red, before executor or main correction:

```text
python -B -m unittest discover -s tests -p test_plan_a_executor.py -v
Ran 8 tests in 0.059s
FAILED (failures=8)
Errors: 0; import/collection errors: 0
```

Cases 1–7 failed explicit requirement assertions that the implementation file
must exist, not imports/collection errors or dummy unconditional failures. Case 8
found two actual API_KEY slices at preimage line 1800. Red was promptly posted to
root board as `board_10f29a87-9c2d-4323-ba4c-3e0fd4e8ed26`. Frozen tests were not
changed after Red. Syntax-only AST parse of all three Python files passed during
the bundled correction; no application import/network or extra suite invocation.

Actual observed Green, after the one bundled implementation correction, using
the identical command:

```text
python -B -m unittest discover -s tests -p test_plan_a_executor.py -v
Ran 8 tests in 0.238s
OK — 8 PASS; failures: 0; errors: 0
```

Exactly **two full local suite runs total**: the honest Red above and this Green.
No subsequent suite/smoke, software/test edit, refactor or bridge implementation
was performed. This brief-only closeout amendment records the already observed
result; it is not another Green invocation or material correction cycle 2.
Local Green covers fakes/algorithm/static startup block only; it does not qualify
Windows custody, live HTTP preservation, native acknowledgement/activation,
logging, protected `/_tools` acceptance or full live DC Done.

## Non-material closeout amendment — 2026-10-26

Direct DC authorized only this status/evidence reconciliation in the existing
brief. The other three software/test files and their Green-tested bytes remain
frozen. Root independently retrieved the initial four artifacts and matched all
four full-file SHA-256 values at `2026-10-26T15:39:39Z`, recorded on board as
`board_77a335c4-eb20-4f3a-a5ce-9b6f0fdee072`. This amendment changes only the brief's
hash; revised-brief retrieval is a distinct metadata readback, not a test rerun.

**FULL OUTCOME NOT MET / NOT RUNTIME-READY.** Pending: actual private JWT,
deployed/reviewed source binding, source-trigger facts, qualified logging,
complete approved selector cohort, qualified existing private custody, interpreted
native acknowledgement, and a connected/qualified owner-mediated activation
bridge with the required protected `/_tools` probe. No successor was generated.
Source-publication proposal, writer/exclusive-write locks, role ownership and
direct DC bounds remain unchanged; no source-control action, extra approval gate
or additional execution authority is created. The original checkpoint remains
`2026-10-26T16:03:24Z`, without extension.
