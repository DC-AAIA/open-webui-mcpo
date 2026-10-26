# Plan A MCPO private executor — amended execution brief

## Current follow-up cycle — direct DC checkpoint disposition

On 2026-10-26 DC answered the actual native checkpoint **"Resume one bounded
cycle"** and directly authorized one material completion cycle. This section is
the current execution record and supersedes the retained first-cycle statements
below concerning disabled CLI, missing protected probe, exit-code-only native
acknowledgement and the old clock. Historical outcomes/provenance remain historical.

Clock: **2026-10-26T17:38:53Z–18:23:53Z**, aim code/Green by **18:03Z**.
Preserved candidate: `6ad35471087bc93dafe0eefffdd559dcdcf7a99b`, exact branch
`plan-a/mcpo-executor-2026-10-25`, clean SA-03 preflight. Same four-file maximum;
only brief, executor and existing test group change this cycle. Application
`src/mcpo/main.py` is unchanged from its already approved seven-line startup
fingerprint-log removal; no other application behavior changes.

Full DC Done remains **usable private executor + approved one real cohort
replacement + protected changed-edge `/_tools` acceptance**. No real replacement
or protected live acceptance occurred during authoring, so **FULL LIVE OUTCOME:
NOT MET**. Source implementation/local fake Green is distinct from runtime Done.

G0 classification remains **accidental-drift assurance**; control-3 scenario is
only trusted-operator wrong-target/unrelated-settings mistakes. Logging/retry
tests are safeguards, not expanded assurance scenarios. One cycle means freeze
the expanded existing fixture group/fake-boundary updates, honest Red, one bundled
software correction, one final Green; at most two full runs this follow-up. Failed
Green means STOP, no software/test edit or rerun. Only truthful brief-only observed
result reconciliation may follow Green. No review agent, installation/helper,
schema/agent/provider/credential-cohort expansion, reset/stash, Git write,
publication, deployment, live credential use, custody setup or real key generation.
Root owns clock checkpoints; no implicit extension or extra approval gate.

### Concrete native interface and source baseline

SA-02 now qualifies CLI **4.40.0** argv/input/output using exact-tag `variable.rs`,
`redeploy.rs`, `service.rs`, `status.rs`, `queries/strings/Project.graphql`,
`mutations/strings/DeploymentRedeploy.graphql`, and `gql/schema.json`.
Official sources:

- <https://raw.githubusercontent.com/railwayapp/cli/v4.40.0/src/commands/variable.rs>
- <https://raw.githubusercontent.com/railwayapp/cli/v4.40.0/src/commands/redeploy.rs>
- <https://raw.githubusercontent.com/railwayapp/cli/v4.40.0/src/gql/mutations/strings/DeploymentRedeploy.graphql>
- <https://raw.githubusercontent.com/railwayapp/cli/v4.40.0/src/gql/queries/strings/Project.graphql>
- <https://raw.githubusercontent.com/railwayapp/cli/v4.40.0/src/gql/schema.json>

SA-04 performed one focused official Project query read, no actual runtime call.
The query returns all project environments, not selected linked environment;
`source.repo/image` are typed, commit/branch/auto-trigger/workspace UUID are not.
Do not guess `meta.commitHash` or infer runtime source from local Git. Native
whoami workspace membership is not proof of project-workspace association.
Retained actual whoami shape is `{name,email,workspaces:[{id,name},...]}` with
no required top-level user id. Identical duplicate workspace memberships from
native aggregation are accepted; inconsistent matched names fail. Project
workspace name is cross-checked, while UUID association remains owner-confirmed.
SA-02's current private native preflight at approximately 17:51Z matched target
project/env/service and linked/explicit status at D0 `c527e170-5719-4d84-96c4-b747655791ab`,
SUCCESS/not-stopped/canRedeploy. It does NOT prove reviewed patched S/branch/trigger
or exclusive window; those owner facts remain unconfirmed. SA-04 performed no
live call and does not convert this metadata result into protected acceptance.

Root/SA-02 subsequently qualified the unmodified frozen Python `NativeRunner`
itself, not only PowerShell's launcher: exact `--version` and explicit-environment
service status passed through its real subprocess/JSON boundary. The service
remained D0 `c527e170-5719-4d84-96c4-b747655791ab`, SUCCESS/not stopped. This was
two read-only client-qualification calls, not another full suite or acceptance
probe. No variable write, deployment or private credential GET occurred.

All native calls use fixed cwd `C:\Projects\open-webui-mcpo`, `shell=False`,
private transient stdout/stderr capture and finite subprocess timeouts. No raw
result, error text or traceback is emitted, logged or written to a public receipt.
No authentication-store, variables/config export, environment or log read.

1. Require version `railway 4.40.0`; privately parse `whoami --json` identity and
   expected workspace membership, then `status --json` expected project and unique
   accessible expected env/service node, GitHub source repo, D0 SUCCESS,
   `deploymentStopped=false`, `canRedeploy=true`.
2. Compare `service status --service <fixed-service> --json` against the same
   vector with `--environment <fixed-env>`. Both exact five-field records
   `id,name,deploymentId,status,stopped` must agree and equal expected-env D0.
   This binds what redeploy uses without relinking. Owner confirms Production
   linkage and current S-to-D0 source binding; metadata cannot manufacture it.
3. Stage exactly `railway variable set API_KEY --stdin --skip-deploys --service
   <fixed-service> --environment <fixed-env> --json`; successor is stdin only.
   Require exit 0 and one exact object `{"keys":["API_KEY"],"set":true}`.
   Reject duplicate keys, trailing data, wrong fields/types or malformed output.
   Missing/uncertain acknowledgement after write intent is UNKNOWN_EFFECTS;
   acknowledgement is not readiness.
4. Revalidate latest/context still D0 immediately before one
   `railway redeploy --service <fixed-service> --yes --json`. No project/env/source/
   deployment-ID flag exists. Source proves it selects latest.id and submits
   deploymentRedeploy; require exact `{"id":"<D1>"}`, canonical UUID, D1 != D0.
5. Poll only explicit-env service status, at most ten attempts, 180-second
   scheduling deadline and 20-second per-call timeout. Require D1 SUCCESS and
   stopped=false; different deployment, malformed output, failure or timeout
   stops without another redeploy, rollback or blind replay.

Stage context is also rechecked immediately before native write. Existing owner
exclusive activation/write window is required: these comparisons are NOT CAS.
Source/schema do not independently establish server variable freshness; actual
stored-key protected acceptance is the required proof. GitHub **Deploy Latest
Commit** remains ONE-TIME code-patch rollout only when needed, current key
unchanged. Routine native redeploy occurs only after current reviewed patched S
is bound to D0. No upload, restart-old, image/registry switch or duplicate trigger.

### Connected private CLI and exact current owner facts

`main` now connects closed public validation -> private owner confirmation ->
NativeActivation native context preflight -> existing custody-leaf preflight ->
current admin JWT no-echo prompt -> PrivateHTTP/NativeBackend -> `execute`.
There is no unconditional disabled-CLI path. Missing actual prerequisites hold
conditionally with sanitized codes before secret mutation; flags alone are not
evidence. Root must not invoke the real CLI until actual facts below are qualified.

Owner enters one bounded JSON object privately in a local terminal, no secret
values. Exact keys (no extras) are the object below; S, D0, Q and selector array
must exactly equal approved public arguments/current evidence, not examples:

```text
{"expected_target":{"project":"62cb5067-2ad6-47b2-97d6-de206f37d18a","environment":"565c4c1e-0616-47d2-8fe3-faa6837fbe65","service":"66391175-bd78-456f-93d5-641498dabcf9"},"expected_source":"S","expected_deployment":"D0","selectors":[{"url":"https://open-webui-mcpo-production.up.railway.app","path":"APPROVED_PATH","info_id":"APPROVED_ID"}],"qualification_reference":"Q","repository":"DC-AAIA/open-webui-mcpo","branch":"main","workspace":"0d9b0ead-cbd9-453d-b8e3-8a0caefa7c81","production_linked":true,"patched_source_active":true,"logging_protected":true,"exclusive_window":true,"complete_cohort":true,"private_admin_auth":true,"leaf_custody_qualified":true}
```

The booleans mean actual current owner-confirmed facts, not permission to invent
them. Source S must already be the active patched source for D0 at approved repo/
main; project/workspace association, Production linkage, complete approved cohort,
logging protection, private current admin auth and qualified leaf must be real.
The owner is responsible for retaining the qualification reference outside public
secret-bearing output. Do not paste a JWT into this object.

Only after those facts/native checks and existing-leaf check does the CLI call
`private_jwt()` in the private local terminal. JWT is entered solely at getpass,
never through model/tools, CLI/env arguments, token store, browser or login.
Non-TTY/echo fallback fails closed. No transcript, tracing or public raw output.

Exact private command grammar (not executed in authoring). Variables are already
qualified NONSECRET metadata; repeat `--selector` for the entire approved cohort:

```powershell
python -B tools/plan_a_executor.py rotate-mcpo --operation-id $OperationId --approval-reference DC-2026-10-26-AMENDED-PACKET --project 62cb5067-2ad6-47b2-97d6-de206f37d18a --environment 565c4c1e-0616-47d2-8fe3-faa6837fbe65 --service 66391175-bd78-456f-93d5-641498dabcf9 --expected-source $ReviewedPatchedSourceSha --expected-deployment $CurrentD0 --selector https://open-webui-mcpo-production.up.railway.app $ApprovedPath $ApprovedInfoId --phase execute --qualification-reference $QualificationReference --logging-qualified --patched-source-qualified --exclusive-writer --auth-available
```

### Protected completion, preservation and custody

Private authenticated config GET, exact cohort/auth checks and fake-sentinel
custody readiness precede any real `secrets.token_hex(32)` generation. Existing
preserve/modify/POST and exact private full-preimage re-GET under exclusive writer
window remain. Only documented type/headers defaults are normalized. After POST,
compare preservation against FINAL private config readback and select the cohort
again. Write PROBE_INTENT, then GET fixed
`https://open-webui-mcpo-production.up.railway.app/_tools` with bearer key obtained
from that final stored readback entry, **not JWT or separately retained successor**.
Require HTTP 200, default TLS, no redirects, bounded private body and no body
logging/count assertions. Only then checkpoint COMPLETE. Failure/unknown probe
leaves non-COMPLETE state and stops; resume cannot replay an unknown intent.
No old-secret test, C1 reopening or n8n/backend requalification is included.

Existing leaf `%LOCALAPPDATA%\DC\PlanA\MCPO` is a separate setup prerequisite.
The code cannot create it or modify ACLs. Owner/private trustee DACL checks apply
to LEAF and state/lock/pending files; ancestors get integrity/no-reparse checks
only. An ancestor's unrelated read-only ACE does NOT require parent ACL stripping
or block an otherwise private leaf. Missing leaf conditionally yields
CUSTODY_UNAVAILABLE before writes if absent or unqualified. Root reports the
one-time leaf setup is now complete under the original approved scope: existing
leaf owner=current user, protected private DACL allows current user/SYSTEM/
Administrators only; ancestor SDDL unchanged, unrelated ancestor read-only ACE
retained. No lock/blob/other state file or successor was created by that setup.
SA-04 did not perform setup or permission changes. Historical .NET DPAPI
synthetic=True and private-leaf success alone did not qualify Python readiness.
Root subsequently ran the actual frozen Python CurrentUser custody adapter with
only its fixed synthetic sentinel: leaf preflight, lock and DPAPI readiness all
passed. One bounded lock for `plan-a-20261026-001` remains; no encrypted operation
blob, pending file or real successor was created. Existing ancestor ACLs remain
unchanged. This is actual host readiness, not a real-key persistence, crash or
cross-machine recovery test. The frozen suite still uses fake OS boundaries.
Existing <=64-KiB CurrentUser protected blob, atomic+flush, lock, same 256-bit
successor, no predecessor/JWT/preimage archives and retained uncertain state/no
cleanup remain. No memory-zeroization or portable recovery claim.

Logging stays unchanged: actual active Redis value-refresh INFO exposure requires
source-backed existing control or conditionally necessary supported WARNING while
preserving ORIGINAL audit policy/delivery; inactive refresh requires no new filter.
No AUDIT_NONE, CRITICAL, invented knob or automatic INFO restoration.

### Follow-up frozen tests and result record

Historical cycle-1 eight Green is not a new PASS. Existing assertions retain the
same scenario. Expanded existing group adds cases 9–13: actual `main` and fixed
native subprocess vectors/owner input; strict stage/activation acknowledgements;
context/noecho/no-early-key generation; final-stored-readback probe credential;
unknown probe effects; actual PrivateHTTP fixed-origin/bearer/HTTP200 behavior.
All added cases/fake-boundary changes were frozen before software correction.

Follow-up Red, actual observed before correction:

```text
python -B -m unittest discover -s tests -p test_plan_a_executor.py -v
Ran 13 tests in 0.268s
FAILED (failures=9); errors/collection errors=0
```

Original eight retained methods passed; new requirements produced nine assertion
failures including subcases. No import error or dummy unconditional failure.
Red board receipt: `board_ebeb5847-18a9-4478-a112-445aa8ad4477`.
Syntax-only AST inspection passed during the one bundle. Final Green is not yet
observed at the pre-Green source freeze; the permitted brief-only closeout now
records the actual final result below, with no software/test edit or rerun:

```text
python -B -m unittest discover -s tests -p test_plan_a_executor.py -v
Ran 13 tests in 0.409s
OK — 13 PASS; failures=0; errors=0
```

Exactly **two full suite runs this follow-up**, one bundled material correction.
The original eight-case historical Green remains separate evidence. No post-Green
software/test change, refactor, additional suite/smoke or live private invocation.
Actual `main` success/ordering, strict native acknowledgements, conditional guards,
stored-readback-key probe and unknown probe effects were exercised with fake
process/input/HTTP/OS custody boundaries, not real Railway/DPAPI/authentication.
Green was reported as `board_2624a16e-f46d-483e-94b8-24ac35695483`.

**Source workflow implementation: CONNECTED / LOCAL GREEN.** The previous missing
CLI, native acknowledgement and protected-probe code gaps are corrected. Remaining
external prerequisites: actual active patched-source S/D0/default-main/trigger and
workspace association confirmation; qualified logging/exclusive window; complete
approved selectors; current private admin JWT; and actual Python CurrentUser
real-successor persistence/operation acceptance. Root separately completed the
private leaf and actual Python synthetic readiness, with ancestors unchanged;
this code authoring created no leaf, ACL or state and generated no successor.
Full live Done remains
NOT MET until the approved actual cohort replacement and protected `/_tools`
acceptance. No live acceptance or production readiness is inferred from fake Green.

Feature-branch preservation and an explicitly incomplete PR may proceed through
SA-03 under the existing approval after root Green. Main merge, code rollout and
credential cutover remain held on their actual source/runtime preconditions.
SA-03's merge proposal remains one code event for the same four files,
pause/restore original active `publish.yaml` ID182507354 and `build-release.yml`
ID182507351; no workflow file/protection/source cleanup changes. No automatic
publication authority comes from this source completion alone. C1 remains closed
positive PASS / old-negative UNTESTED; Plan B parked; aFPM paused. Full live Done
still needs the approved real cohort replacement and actual protected acceptance.

## Retained first-cycle brief (historical, superseded where noted above)

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
