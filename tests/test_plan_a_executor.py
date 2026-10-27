"""Frozen DC eight-case acceptance group. No application imports or live adapters."""
import ast
import contextlib
import copy
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
                    owner_missing=False, guard_missing=False, probe_fail=False):
        req = request()
        d0 = req["expected_deployment"]
        d1 = "bbbbbbbb-1111-2222-3333-bbbbbbbbbbbb"
        workspace = "0d9b0ead-cbd9-453d-b8e3-8a0caefa7c81"
        events, calls = [], []
        backend, custody = PrivateBackend(), MemoryCustody()
        backend.probe_fail = probe_fail
        owner = {"expected_target": req["expected_target"], "expected_source": req["expected_source"],
                 "expected_deployment": d0, "selectors": req["selectors"],
                 "qualification_reference": req["prerequisites"]["qualification_reference"],
                 "repository": "DC-AAIA/open-webui-mcpo", "branch": "main", "workspace": workspace,
                 "production_linked": True, "patched_source_active": True, "logging_protected": True,
                 "exclusive_window": True, "complete_cohort": True, "private_admin_auth": True,
                 "leaf_custody_qualified": True}
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
        def generate(size):
            self.assertEqual(size, 32)
            self.assertIn("get", events)
            self.assertGreater(custody.ready_calls, 0)
            events.append("generate"); return NEW
        argv = ["rotate-mcpo", "--operation-id", req["operation_id"], "--approval-reference", req["approval_reference"],
            "--project", req["expected_target"]["project"], "--environment", req["expected_target"]["environment"],
            "--service", req["expected_target"]["service"], "--expected-source", req["expected_source"],
            "--expected-deployment", d0, "--qualification-reference", req["prerequisites"]["qualification_reference"],
            "--selector", ORIGIN, "openapi.json", "mcpo", "--phase", "execute",
            "--logging-qualified", "--patched-source-qualified", "--exclusive-writer", "--auth-available"]
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(m.subprocess, "run", side_effect=process))
            stack.enter_context(mock.patch.object(m, "private_jwt", side_effect=jwt))
            stack.enter_context(mock.patch.object(m, "PrivateHTTP", return_value=backend))
            stack.enter_context(mock.patch.object(m, "WindowsCustody", return_value=custody))
            # Private nonsecret terminal input only, never an approval CLI flag.
            stack.enter_context(mock.patch.object(m.sys.stdin, "isatty", return_value=True))
            stack.enter_context(mock.patch.object(m.sys.stderr, "isatty", return_value=True))
            stack.enter_context(mock.patch.object(m.sys.stdin, "readline", return_value=json.dumps(owner) + "\n"))
            stack.enter_context(mock.patch.object(m.secrets, "token_hex", side_effect=generate))
            stack.enter_context(contextlib.redirect_stdout(output))
            exitcode = m.main(argv)
        result = json.loads(output.getvalue())
        for sentinel in (JWT, OLD, NEW, "FAKE_PRIVATE_STDERR"):
            self.assertNotIn(sentinel, output.getvalue())
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
