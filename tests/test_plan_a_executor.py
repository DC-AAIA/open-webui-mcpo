"""Frozen DC eight-case acceptance group. No application imports or live adapters."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "https://open-webui-mcpo-production.up.railway.app"
OLD = "FAKE_PREDECESSOR_SENTINEL_ONLY"
NEW = "FAKE_SUCCESSOR_SENTINEL_ONLY"
JWT = "FAKE_ADMIN_JWT_SENTINEL_ONLY"
ADMIN_API_KEY = "sk-FAKE_EXISTING_ADMIN_API_KEY_SENTINEL_ONLY"


def request():
    return {
        "operation_id": "plan-a-test-001",
        "approval_reference": "DC-2026-10-26-AMENDED-PACKET",
        "expected_target": {
            "project": "62cb5067-2ad6-47b2-97d6-de206f37d18a",
            "environment": "565c4c1e-0616-47d2-8fe3-faa6837fbe65",
            "service": "66391175-bd78-456f-93d5-641498dabcf9",
        },
        "expected_source": "a" * 40,
        "expected_deployment": "c527e170-5719-4d84-96c4-b747655791ab",
        "selectors": [{"url": ORIGIN, "path": "openapi.json", "info_id": "mcpo"}],
        "phase": "execute",
        "prerequisites": {
            "logging_qualified": True, "patched_source_qualified": True,
            "exclusive_writer": True, "auth_available": True,
            "qualification_reference": "SA02-PRIVATE-QUALIFICATION-001",
        },
    }


def config():
    return {"TOOL_SERVER_CONNECTIONS": [
        {"url": "https://unrelated.example", "path": "other.json", "key": "OTHER",
         "auth_type": "bearer", "info": {"id": "other", "nested": [3, {"keep": True}]},
         "unknown": {"order": [2, 1]}},
        {"url": ORIGIN, "path": "openapi.json", "key": OLD,
         "auth_type": "bearer", "info": {"id": "mcpo", "name": "keep"},
         "unknown": [1, {"nested": "preserve"}]},
    ], "ENABLE_DIRECT_CONNECTIONS": False, "unknown_top": {"preserve": [3, 2, 1]}}


class MemoryCustody:
    def __init__(self):
        self.record = None
        self.writes = 0
        self.ready_calls = 0
        self.fail = False

    def preflight(self):
        if self.fail:
            raise OSError("FAKE_CUSTODY_FAILURE")

    @contextlib.contextmanager
    def locked(self):
        yield self

    def ready(self):
        self.ready_calls += 1
        if self.fail:
            raise OSError(OLD + JWT)

    def load(self):
        return copy.deepcopy(self.record)

    def save(self, value):
        if self.fail:
            raise OSError(NEW)
        self.record = copy.deepcopy(value)
        self.writes += 1


class PrivateBackend:
    def __init__(self):
        self.value = config()
        self.gets = 0
        self.posts = 0
        self.writes = 0
        self.secrets = []
        self.auth_fail = False
        self.write_fail = False
        self.post_fail = False
        self.concurrent = False
        self.probes = []
        self.probe_fail = False

    def get_config(self):
        self.gets += 1
        if self.auth_fail:
            raise PermissionError(JWT + OLD)
        result = copy.deepcopy(self.value)
        if self.concurrent and self.gets > 1:
            result["unknown_top"]["concurrent"] = True
        return result

    def write_secret(self, successor):
        self.writes += 1
        self.secrets.append(successor)
        if self.write_fail:
            raise TimeoutError(NEW + JWT)

    def post_config(self, value):
        self.posts += 1
        self.value = copy.deepcopy(value)
        if self.post_fail:
            raise TimeoutError(NEW + OLD)

    def probe_tools(self, key):
        self.probes.append(key)
        if self.probe_fail:
            raise TimeoutError("FAKE_PROBE_UNKNOWN")


class FakeActivation:
    """Injected simulation only; not a native Railway qualification."""
    def __init__(self):
        self.calls = 0
        self.qualified = True

    def preflight(self, req):
        if not self.qualified:
            raise RuntimeError("native activation owner required " + OLD)

    def activate(self, req):
        self.calls += 1
        return {"source": req["expected_source"], "deployment": "b" * 8 + "-1111-2222-3333-" + "b" * 12,
                "project": req["expected_target"]["project"],
                "environment": req["expected_target"]["environment"],
                "service": req["expected_target"]["service"], "status": "SUCCESS"}


class PlanAExecutorEightCases(unittest.TestCase):
    def implementation(self):
        path = ROOT / "tools" / "plan_a_executor.py"
        self.assertTrue(path.is_file(), "Requirement: private executor implementation must exist")
        spec = importlib.util.spec_from_file_location("plan_a_executor_frozen", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertTrue(callable(getattr(module, "execute", None)),
                        "Requirement: closed request executor entrypoint must exist")
        # Every external boundary is blocked before any tested entrypoint call.
        # Individual fixtures replace these guards only with synthetic adapters.
        opener = mock.Mock()
        opener.open.side_effect = AssertionError("UNFAKED_HTTP_BOUNDARY")
        guards = (
            mock.patch.object(module.subprocess, "run", side_effect=AssertionError("UNFAKED_PROCESS_BOUNDARY")),
            mock.patch.object(module.urllib.request, "build_opener", return_value=opener),
            mock.patch.object(module.ssl, "create_default_context", return_value=mock.sentinel.fake_tls_context),
            mock.patch.object(module.getpass, "getpass", side_effect=AssertionError("UNFAKED_INPUT_BOUNDARY")),
            mock.patch.object(module.sys.stdin, "isatty", return_value=False),
            mock.patch.object(module.sys.stderr, "isatty", return_value=False),
            mock.patch.object(module.sys.stdin, "readline", side_effect=AssertionError("UNFAKED_OWNER_BOUNDARY")),
            mock.patch.object(module.os, "name", "nt"),
            mock.patch.object(module, "WindowsCustody", side_effect=AssertionError("UNFAKED_CUSTODY_BOUNDARY")),
            mock.patch.object(module.secrets, "token_hex", side_effect=AssertionError("UNFAKED_RANDOM_BOUNDARY")),
            mock.patch("builtins.input", side_effect=AssertionError("UNFAKED_INPUT_BOUNDARY")),
        )
        for guard in guards:
            guard.start()
            self.addCleanup(guard.stop)
        return module

    def run_case(self, module, req=None, backend=None, custody=None, activation=None):
        backend = backend or PrivateBackend()
        custody = custody or MemoryCustody()
        activation = activation or FakeActivation()
        with mock.patch.object(module.secrets, "token_hex", return_value=NEW) as generate:
            result = module.execute(req or request(), backend, custody, activation)
        return result, backend, custody, activation, generate.call_count

    def test_01_wrong_target_or_approval_rejects_before_writes(self):
        m = self.implementation()
        for field, value in (("approval_reference", "peer-approval"), ("expected_target", {}),
                             ("command", "anything"), ("expected_source", "latest")):
            with self.subTest(field=field):
                req = request()
                req[field] = value
                r, b, c, a, generated = self.run_case(m, req)
                self.assertEqual(r["status"], "REJECTED")
                self.assertEqual((b.gets, b.writes, b.posts, c.writes, a.calls, generated), (0, 0, 0, 0, 0, 0))

    def test_02_selector_auth_cohort_and_prerequisite_rejections(self):
        m = self.implementation()
        variants = []
        for field, value in (("auth_type", "none"), ("key", "Bearer " + OLD),
                             ("headers", {"aUtHoRiZaTiOn": "override"})):
            b = PrivateBackend()
            b.value["TOOL_SERVER_CONNECTIONS"][1][field] = value
            variants.append((request(), b))
        for mode in ("missing", "duplicate", "same-origin", "shared-key", "auth-fail"):
            b = PrivateBackend()
            entries = b.value["TOOL_SERVER_CONNECTIONS"]
            if mode == "missing":
                entries.pop()
            elif mode == "duplicate":
                entries.append(copy.deepcopy(entries[1]))
            elif mode == "same-origin":
                extra = copy.deepcopy(entries[1]); extra["info"]["id"] = "outside"
                entries.append(extra)
            elif mode == "shared-key":
                entries[0]["key"] = OLD
            else:
                b.auth_fail = True
            variants.append((request(), b))
        for name in ("logging_qualified", "patched_source_qualified", "exclusive_writer", "auth_available"):
            req = request(); req["prerequisites"][name] = False
            variants.append((req, PrivateBackend()))
        req = request(); req["selectors"].append(copy.deepcopy(req["selectors"][0]))
        variants.append((req, PrivateBackend()))
        for i, (req, b) in enumerate(variants):
            with self.subTest(variant=i):
                r, b, c, a, generated = self.run_case(m, req, b)
                self.assertEqual(r["status"], "REJECTED")
                self.assertEqual((b.writes, b.posts, c.writes, a.calls, generated), (0, 0, 0, 0, 0))
        a = FakeActivation(); a.qualified = False
        r, b, c, a, generated = self.run_case(m, activation=a)
        self.assertEqual(r["code"], "ACTIVATION_OWNER_REQUIRED")
        self.assertEqual((b.writes, b.posts, c.writes, generated), (0, 0, 0, 0))

    def test_03_full_preservation_and_schema_defaults(self):
        m = self.implementation()
        before = config()
        r, b, c, a, generated = self.run_case(m)
        self.assertEqual(r["status"], "COMPLETE")
        expected = copy.deepcopy(before); expected["TOOL_SERVER_CONNECTIONS"][1]["key"] = NEW
        self.assertEqual(b.value, expected)
        self.assertEqual((generated, b.writes, b.posts, a.calls), (1, 1, 1, 1))
        self.assertEqual(m.normalized(before), m.normalized({**before, "TOOL_SERVER_CONNECTIONS": [
            {**entry, "type": "openapi", "headers": {}} for entry in before["TOOL_SERVER_CONNECTIONS"]]}))
        b = PrivateBackend(); b.concurrent = True
        r, b, c, a, generated = self.run_case(m, backend=b)
        self.assertEqual(r["code"], "PREIMAGE_CHANGED")
        self.assertEqual(b.posts, 0)

    def test_04_secret_sentinels_never_escape(self):
        m = self.implementation()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            b = PrivateBackend(); b.write_fail = True
            r, *_ = self.run_case(m, backend=b)
            with mock.patch.object(m.subprocess, "run", side_effect=OSError(NEW + JWT + OLD)):
                with self.assertRaises(m.PlanAError) as caught:
                    m.NativeSecretWriter().write_secret(NEW)
        public = json.dumps(r) + out.getvalue() + err.getvalue() + str(caught.exception)
        for sentinel in (OLD, NEW, JWT):
            self.assertNotIn(sentinel, public)
        self.assertEqual(set(r), {"status", "code", "phase"})

    def test_05_custody_failure_prevents_generation_and_mutation(self):
        m = self.implementation()
        c = MemoryCustody(); c.fail = True
        r, b, c, a, generated = self.run_case(m, custody=c)
        self.assertEqual(r["code"], "CUSTODY_UNAVAILABLE")
        self.assertEqual((b.writes, b.posts, a.calls, generated), (0, 0, 0, 0))

    def test_06_acknowledged_resume_reuses_same_successor(self):
        m = self.implementation()
        r, b, c, a, generated = self.run_case(m)
        self.assertEqual(r["status"], "COMPLETE")
        req = request(); req["phase"] = "resume"
        r, b, c, a, generated = self.run_case(m, req, b, c, a)
        self.assertEqual(r["status"], "COMPLETE")
        self.assertEqual((generated, b.writes, b.posts, a.calls), (0, 1, 1, 1))
        self.assertEqual(c.record["successor"], NEW)
        self.assertNotIn(OLD, json.dumps(c.record))
        self.assertNotIn(JWT, json.dumps(c.record))

    def test_07_unknown_effects_no_replay_or_stale_rollback(self):
        m = self.implementation()
        for failure in ("write_fail", "post_fail"):
            with self.subTest(failure=failure):
                b = PrivateBackend(); setattr(b, failure, True)
                r, b, c, a, generated = self.run_case(m, backend=b)
                self.assertEqual(r["status"], "STOPPED")
                counts = (b.writes, b.posts, a.calls)
                req = request(); req["phase"] = "resume"
                r, b, c, a, generated = self.run_case(m, req, b, c, a)
                self.assertEqual(r["code"], "UNKNOWN_EFFECTS")
                self.assertEqual((b.writes, b.posts, a.calls), counts)
                self.assertEqual(generated, 0)

    def test_08_actual_startup_source_emits_no_key_fragments(self):
        source = (ROOT / "src" / "mcpo" / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "API_KEY":
                offenders.append(node.lineno)
        self.assertEqual(offenders, [], "Requirement: actual startup block must not slice/log API_KEY fragments")
        self.assertIn('logger.info(" API Key: %s", "Provided" if API_KEY else "Not provided")', source)

    def cli_fixture(self, m, *, stage_ack=None, activate_ack=None, context_bad=False,
                    owner_missing=False, guard_missing=False, probe_fail=False,
                    req=None, value=None, selector_args=None, auth_mode=None,
                    owner_patch=None, actual_prompt=False, credential=JWT, prompt_error=None):
        req = copy.deepcopy(request() if req is None else req)
        d0 = req["expected_deployment"]
        d1 = "bbbbbbbb-1111-2222-3333-bbbbbbbbbbbb"
        workspace = "0d9b0ead-cbd9-453d-b8e3-8a0caefa7c81"
        events, calls = [], []
        backend, custody = PrivateBackend(), MemoryCustody()
        backend.probe_fail = probe_fail
        if value is not None:
            backend.value = copy.deepcopy(value)
        owner = {"expected_target": req["expected_target"], "expected_source": req["expected_source"],
                 "expected_deployment": d0, "selectors": req["selectors"],
                 "qualification_reference": req["prerequisites"]["qualification_reference"],
                 "repository": "DC-AAIA/open-webui-mcpo", "branch": "main", "workspace": workspace,
                 "production_linked": True, "patched_source_active": True, "logging_protected": True,
                  "exclusive_window": True, "complete_cohort": True, "private_admin_auth": True,
                  "leaf_custody_qualified": True}
        if "admin_auth_mode" in req:
            owner["admin_auth_mode"] = req["admin_auth_mode"]
        if owner_patch:
            owner.update(owner_patch)
        if owner_missing:
            owner["patched_source_active"] = False
        deployed = [False]

        def status(deployment):
            return {"id": req["expected_target"]["service"], "name": "open-webui-mcpo",
                    "deploymentId": deployment, "status": "SUCCESS", "stopped": False}

        def process(argv, **kwargs):
            calls.append((list(argv), kwargs))
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["cwd"], r"C:\Projects\open-webui-mcpo")
            self.assertTrue(kwargs["capture_output"])
            self.assertNotIn(JWT, str(argv)); self.assertNotIn(NEW, str(argv))
            self.assertNotIn(ADMIN_API_KEY, str(argv))
            args = argv[1:]
            if args == ["--version"]:
                raw = "railway 4.40.0\n"
            elif args == ["whoami", "--json"]:
                raw = json.dumps({"id": "11111111-1111-1111-1111-111111111111", "name": "Synthetic",
                                  "email": "synthetic@example.invalid", "workspaces": [{"id": workspace, "name": "Synthetic"}]})
            elif args == ["status", "--json"]:
                raw = json.dumps({"id": req["expected_target"]["project"], "workspace": {"name": "Synthetic"},
                    "environments": {"edges": [{"node": {"id": req["expected_target"]["environment"],
                    "canAccess": True, "deletedAt": None, "serviceInstances": {"edges": [{"node": {
                    "serviceId": req["expected_target"]["service"], "environmentId": req["expected_target"]["environment"],
                    "source": {"repo": "DC-AAIA/open-webui-mcpo", "image": None},
                    "latestDeployment": {"id": d0, "status": "SUCCESS", "deploymentStopped": False,
                                         "canRedeploy": True}}}]}}}]}})
            elif args[:2] == ["service", "status"]:
                raw = json.dumps(status(d1 if deployed[0] else d0))
                if context_bad and "--environment" not in args:
                    raw = json.dumps(status("cccccccc-1111-2222-3333-cccccccccccc"))
            elif args[:2] == ["variable", "set"]:
                events.append("stage")
                self.assertEqual(kwargs["input"], NEW.encode("ascii"))
                self.assertEqual(args, ["variable", "set", "API_KEY", "--stdin", "--skip-deploys",
                    "--service", req["expected_target"]["service"], "--environment", req["expected_target"]["environment"], "--json"])
                raw = stage_ack if stage_ack is not None else '{"keys":["API_KEY"],"set":true}'
            elif args[:1] == ["redeploy"]:
                events.append("activate"); deployed[0] = True
                self.assertEqual(args, ["redeploy", "--service", req["expected_target"]["service"], "--yes", "--json"])
                raw = activate_ack if activate_ack is not None else json.dumps({"id": d1})
            else:
                self.fail("Unexpected native command vector")
            return types.SimpleNamespace(returncode=0, stdout=raw.encode(), stderr=b"FAKE_PRIVATE_STDERR")

        original_get, original_post, original_probe = backend.get_config, backend.post_config, backend.probe_tools
        def get():
            events.append("get"); return original_get()
        def post(value):
            events.append("post"); return original_post(value)
        def probe(key):
            events.append("probe"); return original_probe(key)
        backend.get_config, backend.post_config, backend.probe_tools = get, post, probe
        def jwt():
            events.append("jwt")
            if guard_missing:
                raise m.PlanAError("NO_ECHO_REQUIRED")
            return JWT
        prompts = []
        def private_prompt(prompt):
            prompts.append(prompt)
            events.append("api-key" if prompt.startswith("Existing Open WebUI admin API key") else "jwt")
            if prompt_error is not None:
                raise prompt_error
            if guard_missing:
                raise m.PlanAError("NO_ECHO_REQUIRED")
            return credential
        def generate(size):
            self.assertEqual(size, 32)
            self.assertIn("get", events)
            self.assertGreater(custody.ready_calls, 0)
            events.append("generate"); return NEW
        argv = ["rotate-mcpo", "--operation-id", req["operation_id"], "--approval-reference", req["approval_reference"],
            "--project", req["expected_target"]["project"], "--environment", req["expected_target"]["environment"],
            "--service", req["expected_target"]["service"], "--expected-source", req["expected_source"],
            "--expected-deployment", d0, "--qualification-reference", req["prerequisites"]["qualification_reference"]]
        if selector_args is None:
            selector_args = []
            for selector in req["selectors"]:
                if "info_id" in selector:
                    selector_args.extend(["--selector", selector["url"], selector["path"], selector["info_id"]])
                else:
                    selector_args.extend(["--selector-no-id", selector["url"], selector["path"]])
        argv.extend(selector_args)
        if auth_mode is not None:
            argv.extend(["--admin-auth-mode", auth_mode])
        argv.extend(["--phase", req["phase"],
            "--logging-qualified", "--patched-source-qualified", "--exclusive-writer", "--auth-available"]
        )
        output, errors = io.StringIO(), io.StringIO()
        backend.request_seen = None
        original_execute = m.execute
        def observed_execute(actual_req, *args):
            backend.request_seen = copy.deepcopy(actual_req)
            return original_execute(actual_req, *args)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(m.subprocess, "run", side_effect=process))
            if actual_prompt:
                stack.enter_context(mock.patch.object(m.getpass, "getpass", side_effect=private_prompt))
            else:
                stack.enter_context(mock.patch.object(m, "private_jwt", side_effect=jwt))
            stack.enter_context(mock.patch.object(m, "PrivateHTTP", return_value=backend))
            stack.enter_context(mock.patch.object(m, "WindowsCustody", return_value=custody))
            stack.enter_context(mock.patch.object(m, "execute", side_effect=observed_execute))
            # Private nonsecret terminal input only, never an approval CLI flag.
            stack.enter_context(mock.patch.object(m.sys.stdin, "isatty", return_value=True))
            stack.enter_context(mock.patch.object(m.sys.stderr, "isatty", return_value=True))
            stack.enter_context(mock.patch.object(m.sys.stdin, "readline", return_value=json.dumps(owner) + "\n"))
            stack.enter_context(mock.patch.object(m.secrets, "token_hex", side_effect=generate))
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(errors))
            # The production guard observes this redirected stream, not the old one.
            stack.enter_context(mock.patch.object(errors, "isatty", return_value=True))
            exitcode = m.main(argv)
        result = json.loads(output.getvalue())
        for sentinel in (JWT, OLD, NEW, "FAKE_PRIVATE_STDERR"):
            self.assertNotIn(sentinel, output.getvalue())
        for sentinel in (JWT, OLD, NEW, ADMIN_API_KEY, "FAKE_PRIVATE_STDERR"):
            self.assertNotIn(sentinel, output.getvalue() + errors.getvalue())
        backend.owner_input, backend.prompt_calls = owner, prompts
        backend.public_output = output.getvalue() + errors.getvalue()
        return result, exitcode, events, calls, backend, custody

    def test_09_actual_cli_native_order_and_exact_ack_success(self):
        m = self.implementation()
        r, exitcode, events, calls, b, c = self.cli_fixture(m)
        self.assertEqual(r["status"], "COMPLETE", "Actual CLI must connect qualified native workflow")
        self.assertEqual(exitcode, 0)
        self.assertEqual(events, ["jwt", "get", "generate", "stage", "activate", "get", "post", "get", "probe"])
        self.assertEqual(b.probes, [NEW])
        self.assertEqual(sum(call[0][1] == "redeploy" for call in calls), 1)
        self.assertEqual(c.record["phase"], "COMPLETE")

    def test_10_actual_cli_bad_ack_unknown_no_activation(self):
        m = self.implementation()
        for raw in ('{}', '{"keys":["API_KEY"],"set":1}', '{"keys":["API_KEY"],"set":true,"set":true}',
                    '{"keys":["API_KEY"],"set":true} trailing', '{"keys":["OTHER"],"set":true}'):
            with self.subTest(raw=raw):
                r, _, events, _, _, c = self.cli_fixture(m, stage_ack=raw)
                self.assertEqual(r["code"], "UNKNOWN_EFFECTS")
                self.assertNotIn("activate", events)
                self.assertEqual(c.record["phase"], "SECRET_WRITE_INTENT")
        r, _, events, _, _, c = self.cli_fixture(m, activate_ack='{"id":"invalid"}')
        self.assertEqual(r["code"], "UNKNOWN_EFFECTS")
        self.assertNotIn("post", events)
        self.assertEqual(c.record["phase"], "ACTIVATION_INTENT")

    def test_11_actual_cli_missing_owner_context_or_noecho_stops_early(self):
        m = self.implementation()
        for kwargs in ({"owner_missing": True}, {"context_bad": True}, {"guard_missing": True}):
            with self.subTest(guard=kwargs):
                r, _, events, _, b, c = self.cli_fixture(m, **kwargs)
                self.assertNotEqual(r["status"], "COMPLETE")
                self.assertNotIn("generate", events)
                self.assertNotIn("stage", events)
                self.assertEqual(c.writes, 0)
                self.assertEqual(b.posts, 0)
                if "guard_missing" not in kwargs:
                    self.assertNotIn("jwt", events)

    def test_12_probe_uses_stored_readback_credential_and_unknown_stops(self):
        m = self.implementation()
        self.assertTrue(callable(getattr(m.PrivateHTTP, "probe_tools", None)),
                        "Requirement: actual protected fixed-origin /_tools method must exist")
        b = PrivateBackend()
        b.value["TOOL_SERVER_CONNECTIONS"][1]["key"] = OLD
        actual_select = m.select_cohort
        def select(value, selectors):
            selected = actual_select(value, selectors)
            if b.gets == 3:
                value["TOOL_SERVER_CONNECTIONS"][selected[0]]["key"] = "FAKE_STORED_READBACK_KEY"
            return selected
        with mock.patch.object(m, "select_cohort", side_effect=select):
            r, *_ = self.run_case(m, backend=b)
        self.assertEqual(r["status"], "COMPLETE")
        self.assertEqual(b.probes, ["FAKE_STORED_READBACK_KEY"])
        r, _, events, _, b, c = self.cli_fixture(m, probe_fail=True)
        self.assertEqual(r["status"], "STOPPED")
        self.assertEqual(c.record["phase"], "PROBE_INTENT")
        self.assertIn("probe", events)

    def test_13_actual_http_probe_fixed_origin_and_bearer(self):
        m = self.implementation()
        self.assertTrue(callable(getattr(m.PrivateHTTP, "probe_tools", None)), "Protected probe implementation required")
        client = m.PrivateHTTP(JWT)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.geturl.return_value = ORIGIN + "/_tools"
        response.read.return_value = b'{"synthetic":true}'
        opener = mock.Mock(); opener.open.return_value = response
        client._opener = opener
        client.probe_tools("FAKE_STORED_READBACK_KEY")
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, ORIGIN + "/_tools")
        self.assertEqual(req.get_header("Authorization"), "Bearer FAKE_STORED_READBACK_KEY")
        self.assertNotIn(JWT, str(req.headers))
        response.status = 401
        with self.assertRaises(m.PlanAError):
            client.probe_tools("FAKE_STORED_READBACK_KEY")

    def test_14_absolute_spec_url_literal_preservation(self):
        m = self.implementation()
        for path in ("openapi.json", "/openapi.json", "nested/openapi.json", ORIGIN + "/openapi.json"):
            with self.subTest(accepted_path=path):
                req, backend = request(), PrivateBackend()
                req["selectors"][0]["path"] = path
                backend.value["TOOL_SERVER_CONNECTIONS"][1]["path"] = path
                before = copy.deepcopy(backend.value)
                r, b, c, a, generated = self.run_case(m, req, backend)
                self.assertEqual(r["status"], "COMPLETE", "Literal qualified absolute spec URL must be supported")
                expected = copy.deepcopy(before)
                expected["TOOL_SERVER_CONNECTIONS"][1]["key"] = NEW
                self.assertEqual(b.value, expected)
                self.assertEqual(b.value["TOOL_SERVER_CONNECTIONS"][1]["path"], path)
                self.assertEqual((generated, b.writes, b.posts, a.calls), (1, 1, 1, 1))
        for path in ("https://unrelated.example/openapi.json", ORIGIN.replace("https:", "http:") + "/openapi.json",
                     ORIGIN + ":443/openapi.json", ORIGIN.replace("https://", "https://user@") + "/openapi.json",
                     ORIGIN + "/openapi.json?x=1", ORIGIN + "/openapi.json#fragment",
                     ORIGIN + "/%6Fpenapi.json", ORIGIN + "/../openapi.json", ORIGIN + "//openapi.json"):
            with self.subTest(rejected_path=path):
                req = request(); req["selectors"][0]["path"] = path
                r, b, c, a, generated = self.run_case(m, req)
                self.assertEqual(r["status"], "REJECTED")
                self.assertEqual((b.gets, b.writes, b.posts, c.writes, a.calls, generated), (0, 0, 0, 0, 0, 0))

    def test_15_no_explicit_id_selection_and_preservation(self):
        m = self.implementation()
        selector = {"url": ORIGIN, "path": "openapi.json", "no_explicit_id": True}
        for shape in ("missing-info", "missing-id", "null-id", "empty-id"):
            with self.subTest(accepted_shape=shape):
                req, backend = request(), PrivateBackend()
                req["selectors"] = [copy.deepcopy(selector)]
                entry = backend.value["TOOL_SERVER_CONNECTIONS"][1]
                if shape == "missing-info":
                    entry.pop("info")
                elif shape == "missing-id":
                    entry["info"].pop("id")
                else:
                    entry["info"]["id"] = None if shape == "null-id" else ""
                before = copy.deepcopy(backend.value)
                r, b, c, a, generated = self.run_case(m, req, backend)
                self.assertEqual(r["status"], "COMPLETE", "Explicit no-ID selection must preserve the stored shape")
                expected = copy.deepcopy(before); expected["TOOL_SERVER_CONNECTIONS"][1]["key"] = NEW
                self.assertEqual(b.value, expected)
                self.assertEqual((generated, b.writes, b.posts, a.calls), (1, 1, 1, 1))
        variants = []
        for info in (None, [], "", False, {"id": 0}, {"id": False}, {"id": []}, {"id": {}},
                     {"id": "auto"}, {"id": "1"}):
            backend = PrivateBackend(); backend.value["TOOL_SERVER_CONNECTIONS"][1]["info"] = info
            req = request(); req["selectors"] = [copy.deepcopy(selector)]
            variants.append((req, backend))
        for extra in ({"no_explicit_id": False}, {"no_explicit_id": 1}, {"info_id": "mcpo"}, {"extra": True}):
            req = request(); req["selectors"] = [{**selector, **extra}]
            variants.append((req, PrivateBackend()))
        for shape in ("ambiguous", "incomplete"):
            backend = PrivateBackend()
            entry = backend.value["TOOL_SERVER_CONNECTIONS"][1]; entry["info"].pop("id")
            other = copy.deepcopy(entry)
            if shape == "incomplete":
                other["path"] = "/outside/openapi.json"
            backend.value["TOOL_SERVER_CONNECTIONS"].append(other)
            req = request(); req["selectors"] = [copy.deepcopy(selector)]
            variants.append((req, backend))
        req = request(); req["selectors"] = [copy.deepcopy(selector), copy.deepcopy(selector)]
        variants.append((req, PrivateBackend()))
        for index, (req, backend) in enumerate(variants):
            with self.subTest(rejected_variant=index):
                r, b, c, a, generated = self.run_case(m, req, backend)
                self.assertEqual(r["status"], "REJECTED")
                self.assertEqual((b.writes, b.posts, c.writes, a.calls, generated), (0, 0, 0, 0, 0))
        with self.subTest(repeated_match_index=True):
            explicit = request()["selectors"][0]
            with self.assertRaises(m.PlanAError):
                m.select_cohort(config(), [explicit, copy.deepcopy(explicit)])
        with self.subTest(literal_auto_requires_actual_stored_id=True):
            req, backend = request(), PrivateBackend()
            req["selectors"][0]["info_id"] = "auto"
            backend.value["TOOL_SERVER_CONNECTIONS"][1]["info"]["id"] = "auto"
            r, *_ = self.run_case(m, req, backend)
            self.assertEqual(r["status"], "COMPLETE")
        with self.subTest(derived_index_is_not_stored_id=True):
            req, backend = request(), PrivateBackend()
            req["selectors"][0]["info_id"] = "1"
            backend.value["TOOL_SERVER_CONNECTIONS"][1]["info"].pop("id")
            r, b, c, a, generated = self.run_case(m, req, backend)
            self.assertEqual(r["status"], "REJECTED")
            self.assertEqual((generated, b.writes, b.posts), (0, 0, 0))

    def test_16_cli_selector_forms_and_legacy_compatibility(self):
        m = self.implementation()
        req, value = request(), config()
        req["selectors"] = [
            {"url": ORIGIN, "path": "openapi.json", "info_id": "mcpo"},
            {"url": ORIGIN, "path": "/named/openapi.json", "info_id": "second"},
            {"url": ORIGIN, "path": ORIGIN + "/openapi.json", "no_explicit_id": True},
            {"url": ORIGIN, "path": "blank/openapi.json", "no_explicit_id": True},
        ]
        for selector in req["selectors"][1:]:
            entry = copy.deepcopy(value["TOOL_SERVER_CONNECTIONS"][1])
            entry["path"] = selector["path"]
            if "info_id" in selector:
                entry["info"]["id"] = selector["info_id"]
            else:
                entry["info"].pop("id")
            value["TOOL_SERVER_CONNECTIONS"].append(entry)
        interleaved = ["--selector-no-id", ORIGIN, ORIGIN + "/openapi.json",
                       "--selector", ORIGIN, "openapi.json", "mcpo",
                       "--selector-no-id", ORIGIN, "blank/openapi.json",
                       "--selector", ORIGIN, "/named/openapi.json", "second"]
        with self.subTest(mixed_repeatable_flags=True):
            r, exitcode, events, calls, b, c = self.cli_fixture(m, req=req, value=value, selector_args=interleaved)
            self.assertEqual(r["status"], "COMPLETE", "Actual closed CLI must accept repeatable two-argument no-ID selectors")
            self.assertEqual(exitcode, 0)
            self.assertEqual(b.request_seen, req)
            self.assertEqual(b.owner_input["selectors"], req["selectors"])
            self.assertEqual([e["key"] for e in b.value["TOOL_SERVER_CONNECTIONS"]], ["OTHER", NEW, NEW, NEW, NEW])
        with self.subTest(legacy_cli=True):
            r, exitcode, events, calls, b, c = self.cli_fixture(m)
            self.assertEqual((r["status"], exitcode), ("COMPLETE", 0))
            self.assertEqual(b.request_seen, request())
        invalid_args = [[], ["--selector-no-id", ORIGIN], ["--selector", ORIGIN, "openapi.json"],
                        ["--selector", ORIGIN, "openapi.json", "mcpo"] * 2,
                        ["--selector", ORIGIN, "openapi.json", "mcpo"] * 33]
        for args in invalid_args:
            with self.subTest(invalid_argv=args):
                r, exitcode, events, calls, b, c = self.cli_fixture(m, selector_args=args)
                self.assertEqual((r["code"], exitcode), ("INVALID_REQUEST", 2))
                self.assertEqual((events, calls, b.posts, c.writes), ([], [], 0, 0))
        with self.subTest(combined_count_limit=True):
            req = request()
            req["selectors"] = [{"url": ORIGIN, "path": "openapi.json", "info_id": "server" + str(i)} for i in range(32)]
            m.validate_request(req)
            for selectors in ([], req["selectors"] + [{"url": ORIGIN, "path": "openapi.json", "info_id": "extra"}]):
                invalid = copy.deepcopy(req); invalid["selectors"] = selectors
                with self.assertRaises(m.PlanAError):
                    m.validate_request(invalid)

    def test_17_existing_api_key_mode_owner_and_noecho(self):
        m = self.implementation()
        api_req = request(); api_req["admin_auth_mode"] = "existing-api-key"
        jwt_prompt = "Existing current Open WebUI admin JWT (private, no echo): "
        api_prompt = "Existing Open WebUI admin API key (private, no echo; existing key only): "
        prohibited = {"existing_admin_api_key_qualified", "api_key_feature_enabled", "api_key_config_route_permitted"}
        with self.subTest(explicit_api_key_mode=True):
            r, exitcode, events, calls, b, c = self.cli_fixture(
                m, req=api_req, auth_mode="existing-api-key", actual_prompt=True, credential=ADMIN_API_KEY)
            self.assertEqual((r["status"], exitcode), ("COMPLETE", 0), "Explicit existing-key mode must reach the fake engine")
            self.assertEqual(b.request_seen, api_req)
            self.assertEqual(b.prompt_calls, [api_prompt])
            self.assertEqual(events, ["api-key", "get", "generate", "stage", "activate", "get", "post", "get", "probe"])
            self.assertEqual(set(b.owner_input), set(m.owner_contract(request())) | {"admin_auth_mode"})
            self.assertTrue(prohibited.isdisjoint(b.owner_input))
            self.assertEqual(b.owner_input, m.owner_contract(api_req))
            self.assertEqual(b.probes, [NEW])
        for mode in (None, "jwt"):
            with self.subTest(legacy_jwt_mode=mode):
                r, exitcode, events, calls, b, c = self.cli_fixture(m, auth_mode=mode, actual_prompt=True)
                self.assertEqual((r["status"], exitcode), ("COMPLETE", 0))
                self.assertEqual(b.request_seen, request())
                self.assertNotIn("admin_auth_mode", b.owner_input)
                self.assertEqual(b.prompt_calls, [jwt_prompt])
        for req, mode, credential in ((request(), None, ADMIN_API_KEY), (api_req, "existing-api-key", JWT)):
            with self.subTest(cross_mode_credential=mode):
                r, exitcode, events, calls, b, c = self.cli_fixture(
                    m, req=req, auth_mode=mode, actual_prompt=True, credential=credential)
                self.assertEqual((r["code"], exitcode), ("NO_ECHO_REQUIRED", 2))
                self.assertNotIn("get", events)
                self.assertNotIn("generate", events)
                self.assertEqual((b.posts, c.writes), (0, 0))
        for credential, error in (("", None), ("sk-FAKE WITH SPACE", None),
                                  (ADMIN_API_KEY, m.getpass.GetPassWarning("FAKE_NO_ECHO_WARNING"))):
            with self.subTest(noecho_credential=credential, warning=error is not None):
                r, exitcode, events, calls, b, c = self.cli_fixture(
                    m, req=api_req, auth_mode="existing-api-key", actual_prompt=True,
                    credential=credential, prompt_error=error)
                self.assertEqual((r["code"], exitcode), ("NO_ECHO_REQUIRED", 2))
                self.assertNotIn("generate", events)
                self.assertEqual((b.posts, c.writes), (0, 0))
        for patch in ({"admin_auth_mode": None}, {"admin_auth_mode": "jwt"},
                      *({name: True} for name in sorted(prohibited))):
            with self.subTest(owner_mode_or_extra=patch):
                r, exitcode, events, calls, b, c = self.cli_fixture(
                    m, req=api_req, auth_mode="existing-api-key", owner_patch=patch,
                    actual_prompt=True, credential=ADMIN_API_KEY)
                self.assertEqual((r["code"], exitcode), ("SOURCE_OWNER_CONFIRMATION_REQUIRED", 2))
                self.assertEqual((events, calls, b.prompt_calls, b.posts, c.writes), ([], [], [], 0, 0))
        with self.subTest(unsupported_auth_mode=True):
            r, exitcode, events, calls, b, c = self.cli_fixture(m, auth_mode="password")
            self.assertEqual((r["code"], exitcode), ("INVALID_REQUEST", 2))
            self.assertEqual((events, calls, b.posts, c.writes), ([], [], 0, 0))
        for field in ("stdin", "stderr"):
            with self.subTest(non_tty=field):
                with mock.patch.object(getattr(m.sys, field), "isatty", return_value=False):
                    with self.assertRaises(m.PlanAError) as caught:
                        m.private_jwt()
                self.assertEqual(caught.exception.code, "NO_ECHO_REQUIRED")
        with self.subTest(non_windows=True):
            with mock.patch.object(m.os, "name", "posix"):
                with self.assertRaises(m.PlanAError) as caught:
                    m.private_jwt()
            self.assertEqual(caught.exception.code, "NO_ECHO_REQUIRED")

    def test_18_config_http_bearer_and_denial_guards(self):
        m = self.implementation()
        value = {"TOOL_SERVER_CONNECTIONS": copy.deepcopy(config()["TOOL_SERVER_CONNECTIONS"])}
        for entry in value["TOOL_SERVER_CONNECTIONS"]:
            entry["config"] = {}
        def response_for(status=200, url=None, body=None):
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.status = status
            response.geturl.return_value = m.CONFIG_URL if url is None else url
            response.read.return_value = json.dumps(value).encode() if body is None else body
            return response
        for credential in (JWT, ADMIN_API_KEY):
            with self.subTest(config_bearer=credential):
                opener = mock.Mock(); opener.open.return_value = response_for()
                with mock.patch.object(m.urllib.request, "build_opener", return_value=opener):
                    client = m.PrivateHTTP(credential)
                self.assertEqual(client.get_config(), value)
                client.post_config(value)
                requests = [call.args[0] for call in opener.open.call_args_list]
                self.assertEqual([req.get_method() for req in requests], ["GET", "POST"])
                self.assertEqual([req.full_url for req in requests], [m.CONFIG_URL, m.CONFIG_URL])
                self.assertEqual([req.get_header("Authorization") for req in requests], ["Bearer " + credential] * 2)
                self.assertEqual(json.loads(requests[1].data), value)
                self.assertTrue(all(call.kwargs == {"timeout": m.TIMEOUT} for call in opener.open.call_args_list))
        failures = [response_for(status=401), response_for(status=403),
                    response_for(url=ORIGIN + "/redirected"), response_for(body=b'{"TOOL_SERVER_CONNECTIONS":'),
                    response_for(body=b"x" * (m.MAX_RESPONSE + 1)),
                    response_for(body=b'{"TOOL_SERVER_CONNECTIONS":[],"extra":true}'),
                    response_for(body=b'{"TOOL_SERVER_CONNECTIONS":[],"TOOL_SERVER_CONNECTIONS":[]}')]
        for index, response in enumerate(failures):
            with self.subTest(initial_private_get_failure=index):
                opener = mock.Mock(); opener.open.return_value = response
                with mock.patch.object(m.urllib.request, "build_opener", return_value=opener):
                    client = m.PrivateHTTP(ADMIN_API_KEY)
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    r, b, c, a, generated = self.run_case(m, backend=client)
                self.assertEqual(r, {"status": "REJECTED", "code": "PRIVATE_READ_FAILED", "phase": "PREFLIGHT"})
                self.assertEqual((generated, c.writes, a.calls, opener.open.call_count), (0, 0, 0, 1))
                self.assertEqual(opener.open.call_args.args[0].get_method(), "GET")
                m.subprocess.run.assert_not_called()
                self.assertEqual(out.getvalue() + err.getvalue(), "")
                for sentinel in (ADMIN_API_KEY, JWT, OLD, NEW):
                    self.assertNotIn(sentinel, json.dumps(r))

    def test_19_binding_resume_and_secret_non_disclosure(self):
        m = self.implementation()
        legacy = request()
        expected = hashlib.sha256(json.dumps({k: v for k, v in legacy.items() if k != "phase"},
                                             sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(m.binding(legacy), expected)
        api_req = copy.deepcopy(legacy); api_req["admin_auth_mode"] = "existing-api-key"
        noid_req = copy.deepcopy(legacy)
        noid_req["selectors"] = [{"url": ORIGIN, "path": "openapi.json", "no_explicit_id": True}]
        absolute_req = copy.deepcopy(legacy); absolute_req["selectors"][0]["path"] = ORIGIN + "/openapi.json"
        for changed in (api_req, noid_req, absolute_req):
            self.assertNotEqual(m.binding(changed), expected)
        resume = copy.deepcopy(legacy); resume["phase"] = "resume"
        self.assertEqual(m.binding(resume), expected)
        left, right = config(), config()
        left["TOOL_SERVER_CONNECTIONS"][1]["info"].pop("id")
        right["TOOL_SERVER_CONNECTIONS"][1]["info"]["id"] = None
        self.assertNotEqual(m.normalized(left), m.normalized(right))
        right = copy.deepcopy(left); right["TOOL_SERVER_CONNECTIONS"][1]["path"] = ORIGIN + "/openapi.json"
        self.assertNotEqual(m.normalized(left), m.normalized(right))
        r, backend, custody, activation, generated = self.run_case(m)
        self.assertEqual(r["status"], "COMPLETE")
        counts = (backend.writes, backend.posts, activation.calls, custody.writes)
        for changed in (api_req, noid_req, absolute_req):
            with self.subTest(changed_binding_resume=changed):
                changed = copy.deepcopy(changed); changed["phase"] = "resume"
                r, b, c, a, generated = self.run_case(m, changed, backend, custody, activation)
                self.assertNotEqual(r["status"], "COMPLETE")
                self.assertEqual((b.writes, b.posts, a.calls, c.writes), counts)
                self.assertEqual(generated, 0)
        for phase in ("SECRET_WRITE_INTENT", "ACTIVATION_INTENT", "CONFIG_POST_INTENT", "PROBE_INTENT"):
            with self.subTest(unknown_intent=phase):
                c = MemoryCustody(); c.record = {"binding": expected, "successor": NEW, "phase": phase}
                r, b, c, a, generated = self.run_case(m, resume, custody=c)
                self.assertEqual(r["code"], "UNKNOWN_EFFECTS")
                self.assertEqual((b.gets, b.writes, b.posts, a.calls, generated), (0, 0, 0, 0, 0))
        with self.subTest(ready_resume_reuses_successor=True):
            c = MemoryCustody(); c.record = {"binding": expected, "successor": NEW, "phase": "READY"}
            r, b, c, a, generated = self.run_case(m, resume, custody=c)
            self.assertEqual(r["status"], "COMPLETE")
            self.assertEqual((generated, b.writes, b.posts, a.calls), (0, 1, 1, 1))
            for sentinel in (ADMIN_API_KEY, JWT, OLD):
                self.assertNotIn(sentinel, json.dumps(c.record))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            client = m.PrivateHTTP(ADMIN_API_KEY)
            client._opener.open.side_effect = OSError(ADMIN_API_KEY + JWT + OLD + NEW)
            with self.assertRaises(m.PlanAError) as caught:
                client.get_config()
        public = out.getvalue() + err.getvalue() + str(caught.exception)
        for sentinel in (ADMIN_API_KEY, JWT, OLD, NEW):
            self.assertNotIn(sentinel, public)


if __name__ == "__main__":
    unittest.main(verbosity=2)
