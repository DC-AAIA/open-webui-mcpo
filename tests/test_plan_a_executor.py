"""Frozen DC eight-case acceptance group. No application imports or live adapters."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
