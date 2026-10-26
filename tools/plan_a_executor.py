"""Bounded Plan A private executor. Stdlib only; never run via model tools.

The public CLI deliberately has NO qualified live activation adapter. It stops at
ACTIVATION_OWNER_REQUIRED before a private GET, key generation or mutation. The
injected engine is acceptance-tested with fakes, not production-qualified.
"""
import argparse
import contextlib
import copy
import ctypes
from ctypes import wintypes
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings

TARGET = {"project": "62cb5067-2ad6-47b2-97d6-de206f37d18a",
          "environment": "565c4c1e-0616-47d2-8fe3-faa6837fbe65",
          "service": "66391175-bd78-456f-93d5-641498dabcf9"}
MCPO_ORIGIN = "https://open-webui-mcpo-production.up.railway.app"
WEBUI_ORIGIN = "https://open-webui-production-b978.up.railway.app"
CONFIG_URL = WEBUI_ORIGIN + "/api/v1/configs/tool_servers"
APPROVAL = "DC-2026-10-26-AMENDED-PACKET"
MAX_BLOB = 65536
MAX_RESPONSE = 1024 * 1024
TIMEOUT = 20
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
CODES = frozenset({"INVALID_REQUEST", "PREREQUISITES_REQUIRED", "SELECTOR_REJECTED",
                   "PRIVATE_READ_FAILED", "CUSTODY_UNAVAILABLE", "ACTIVATION_OWNER_REQUIRED",
                   "UNKNOWN_EFFECTS", "PREIMAGE_CHANGED", "READBACK_MISMATCH",
                   "ACTIVATION_PROOF_INVALID", "OPERATION_EXISTS", "RESUME_REQUIRED",
                   "PRIVATE_TRANSPORT_FAILED", "PRIVATE_WRITE_FAILED", "NO_ECHO_REQUIRED",
                   "COMPLETE", "INTERNAL_STOP"})
PHASES = frozenset({"PREFLIGHT", "READY", "SECRET_WRITE_INTENT", "SECRET_STAGED",
                    "ACTIVATION_INTENT", "ACTIVATED", "CONFIG_POST_INTENT", "COMPLETE"})


class PlanAError(Exception):
    def __init__(self, code):
        self.code = code if code in CODES else "INTERNAL_STOP"
        super().__init__(self.code)


def receipt(status, code, phase="PREFLIGHT"):
    # No interpolation of exceptions, identifiers, headers, bodies or child output.
    return {"status": status if status in {"REJECTED", "STOPPED", "COMPLETE"} else "STOPPED",
            "code": code if code in CODES else "INTERNAL_STOP",
            "phase": phase if phase in PHASES else "PREFLIGHT"}


def _uuid(value):
    return isinstance(value, str) and str(uuid.UUID(value)) == value


def validate_request(req):
    try:
        fields = {"operation_id", "approval_reference", "expected_target", "expected_source",
                  "expected_deployment", "selectors", "phase", "prerequisites"}
        if type(req) is not dict or set(req) != fields:
            raise ValueError
        if not isinstance(req["operation_id"], str) or not SAFE_ID.fullmatch(req["operation_id"]):
            raise ValueError
        if req["approval_reference"] != APPROVAL or req["expected_target"] != TARGET:
            raise ValueError
        if not isinstance(req["expected_source"], str) or not re.fullmatch(r"[0-9a-f]{40}", req["expected_source"]):
            raise ValueError
        if not _uuid(req["expected_deployment"]) or req["phase"] not in {"execute", "resume"}:
            raise ValueError
        selectors = req["selectors"]
        if type(selectors) is not list or not 1 <= len(selectors) <= 32:
            raise ValueError
        tuples = []
        for selector in selectors:
            if type(selector) is not dict or set(selector) != {"url", "path", "info_id"}:
                raise ValueError
            if selector["url"] != MCPO_ORIGIN:
                raise ValueError
            path = selector["path"]
            if not isinstance(path, str) or not re.fullmatch(r"/?[A-Za-z0-9_/-]*openapi\.json", path) or "//" in path:
                raise ValueError
            if not isinstance(selector["info_id"], str) or not SAFE_ID.fullmatch(selector["info_id"]):
                raise ValueError
            tuples.append((selector["url"], path, selector["info_id"]))
        if len(set(tuples)) != len(tuples):
            raise ValueError
        pre = req["prerequisites"]
        flags = {"logging_qualified", "patched_source_qualified", "exclusive_writer", "auth_available"}
        if type(pre) is not dict or set(pre) != flags | {"qualification_reference"}:
            raise ValueError
        if any(pre[k] is not True for k in flags):
            raise PlanAError("PREREQUISITES_REQUIRED")
        if not isinstance(pre["qualification_reference"], str) or not SAFE_ID.fullmatch(pre["qualification_reference"]):
            raise ValueError
    except PlanAError:
        raise
    except Exception:
        raise PlanAError("INVALID_REQUEST") from None


def normalized(value):
    """Only documented per-entry defaults: absent type=openapi, empty headers.

    None/absent/{} headers have the same no-custom-header effect. No unknown
    field removal, list sorting, auth normalization or key coercion is allowed.
    """
    value = copy.deepcopy(value)
    for entry in value["TOOL_SERVER_CONNECTIONS"]:
        entry.setdefault("type", "openapi")
        if entry.get("headers") is None:
            entry["headers"] = {}
    return value


def select_cohort(value, selectors):
    try:
        entries = value["TOOL_SERVER_CONNECTIONS"]
        if type(value) is not dict or type(entries) is not list or len(entries) > 512:
            raise ValueError
        matches = []
        for selector in selectors:
            found = [i for i, e in enumerate(entries) if type(e) is dict
                     and e.get("url") == selector["url"] and e.get("path") == selector["path"]
                     and type(e.get("info")) is dict and e["info"].get("id") == selector["info_id"]]
            if len(found) != 1:
                raise ValueError
            matches.extend(found)
        keys = []
        for i in matches:
            e = entries[i]
            key = e.get("key")
            if e.get("auth_type") != "bearer" or e.get("type", "openapi") != "openapi":
                raise ValueError
            if not isinstance(key, str) or not key or any(c.isspace() for c in key) or key.lower().startswith("bearer"):
                raise ValueError
            headers = e.get("headers")
            if headers is not None and type(headers) is not dict:
                raise ValueError
            if headers and any(not isinstance(k, str) or k.lower() == "authorization" for k in headers):
                raise ValueError
            keys.append(key)
        if len(set(keys)) != 1:
            raise ValueError
        for i, entry in enumerate(entries):
            if type(entry) is not dict:
                raise ValueError
            if i not in matches:
                parsed = urllib.parse.urlsplit(entry.get("url", ""))
                same_origin = parsed.scheme.lower() == "https" and parsed.hostname == urllib.parse.urlsplit(MCPO_ORIGIN).hostname
                extra_headers = entry.get("headers")
                shared_header = type(extra_headers) is dict and any(
                    isinstance(v, str) and (v in keys or any(v == "Bearer " + key for key in keys))
                    for v in extra_headers.values())
                if same_origin or entry.get("key") in keys or shared_header:
                    raise ValueError
        return matches
    except Exception:
        raise PlanAError("SELECTOR_REJECTED") from None


def binding(req):
    return hashlib.sha256(json.dumps({k: v for k, v in req.items() if k != "phase"},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_activation(proof, req):
    try:
        if type(proof) is not dict or set(proof) != {"project", "environment", "service", "source", "deployment", "status"}:
            raise ValueError
        if any(proof[k] != v for k, v in TARGET.items()) or proof["source"] != req["expected_source"]:
            raise ValueError
        if proof["status"] != "SUCCESS" or not _uuid(proof["deployment"]) or proof["deployment"] == req["expected_deployment"]:
            raise ValueError
    except Exception:
        raise PlanAError("ACTIVATION_PROOF_INVALID") from None


def execute(req, backend, custody, activation):
    """Private-only engine; injected adapters are trusted code, never CLI input.

    Write-ahead intents stop uncertain outcomes. No rollback, retry, key printing,
    predecessor persistence or blind replay. COMPLETE resume is acknowledgement,
    not a new live validation. The exclusive-writer window is NOT server CAS.
    """
    phase = "PREFLIGHT"
    effects_possible = False
    try:
        validate_request(req)
        try:
            activation.preflight(req)
        except Exception:
            raise PlanAError("ACTIVATION_OWNER_REQUIRED") from None
        with custody.locked():
            try:
                state = custody.load()
            except Exception:
                raise PlanAError("CUSTODY_UNAVAILABLE") from None
            if state is not None:
                if state.get("binding") != binding(req) or req["phase"] != "resume":
                    raise PlanAError("OPERATION_EXISTS")
                phase = state.get("phase", "PREFLIGHT")
                if phase in {"SECRET_WRITE_INTENT", "ACTIVATION_INTENT", "CONFIG_POST_INTENT"}:
                    return receipt("STOPPED", "UNKNOWN_EFFECTS", phase)
                if phase == "COMPLETE":
                    return receipt("COMPLETE", "COMPLETE", phase)
                # READY is acknowledged custody before any external write. It
                # can safely revalidate a fresh preimage using the same key.
                # Later phases require reconciliation: no preimage is persisted.
                if phase != "READY":
                    return receipt("STOPPED", "UNKNOWN_EFFECTS", phase)
                if not isinstance(state.get("successor"), str) or not state["successor"]:
                    raise PlanAError("CUSTODY_UNAVAILABLE")
            elif req["phase"] != "execute":
                raise PlanAError("RESUME_REQUIRED")
            try:
                before = backend.get_config()
            except Exception:
                raise PlanAError("PRIVATE_READ_FAILED") from None
            selected = select_cohort(before, req["selectors"])
            try:
                custody.ready()  # Fixed fake sentinel; NEVER generates a real key.
            except Exception:
                raise PlanAError("CUSTODY_UNAVAILABLE") from None
            successor = state["successor"] if state is not None else secrets.token_hex(32)
            # Exactly 256 random bits once; acknowledged READY resume never regenerates.
            state = {"binding": binding(req), "successor": successor, "phase": "READY"}

            def checkpoint(next_phase):
                nonlocal phase
                state["phase"] = next_phase
                try:
                    custody.save(state)
                except Exception:
                    raise PlanAError("CUSTODY_UNAVAILABLE") from None
                phase = next_phase

            checkpoint("READY")
            checkpoint("SECRET_WRITE_INTENT")
            effects_possible = True
            backend.write_secret(successor)
            checkpoint("SECRET_STAGED")
            checkpoint("ACTIVATION_INTENT")
            proof = activation.activate(req)
            validate_activation(proof, req)
            state["activation"] = proof  # Nonsecret, exact-target receipt only.
            checkpoint("ACTIVATED")
            # Full private preimage comparison immediately before POST. No CAS.
            current = backend.get_config()
            if current != before:
                raise PlanAError("PREIMAGE_CHANGED")
            after = copy.deepcopy(before)
            for i in selected:
                after["TOOL_SERVER_CONNECTIONS"][i]["key"] = successor
            checkpoint("CONFIG_POST_INTENT")
            backend.post_config(after)
            readback = backend.get_config()
            if normalized(readback) != normalized(after):
                raise PlanAError("READBACK_MISMATCH")
            select_cohort(readback, req["selectors"])
            checkpoint("COMPLETE")
            return receipt("COMPLETE", "COMPLETE", phase)
    except PlanAError as exc:
        return receipt("STOPPED" if effects_possible else "REJECTED", exc.code, phase)
    except BaseException:
        return receipt("STOPPED" if effects_possible else "REJECTED",
                       "UNKNOWN_EFFECTS" if effects_possible else "INTERNAL_STOP", phase)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PlanAError("PRIVATE_TRANSPORT_FAILED")


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanAError("PRIVATE_TRANSPORT_FAILED")
        result[key] = value
    return result


class PrivateHTTP:
    """Fixed-origin private JWT boundary; never exposes response/error text."""
    def __init__(self, jwt):
        if not isinstance(jwt, str) or not jwt or any(c.isspace() for c in jwt):
            raise PlanAError("NO_ECHO_REQUIRED")
        self._jwt = jwt
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                         urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def _call(self, method, value=None):
        try:
            data = None if value is None else json.dumps(value, allow_nan=False).encode("utf-8")
            if data is not None and len(data) > MAX_RESPONSE:
                raise ValueError
            req = urllib.request.Request(CONFIG_URL, data=data, method=method,
                      headers={"Authorization": "Bearer " + self._jwt,
                               "Content-Type": "application/json", "Accept": "application/json"})
            with self._opener.open(req, timeout=TIMEOUT) as response:
                if response.geturl() != CONFIG_URL or response.status != 200:
                    raise ValueError
                raw = response.read(MAX_RESPONSE + 1)
                if len(raw) > MAX_RESPONSE:
                    raise ValueError
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_pairs,
                                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                # The live v0.8.10 envelope has exactly one field. Unknown entry
                # fields are preserved by extra=allow; outer extras aren't safe.
                if type(value) is not dict or set(value) != {"TOOL_SERVER_CONNECTIONS"}:
                    raise ValueError
                entries = value["TOOL_SERVER_CONNECTIONS"]
                if type(entries) is not list or any(type(e) is not dict or
                    not {"url", "path", "auth_type", "key", "config"} <= set(e) for e in entries):
                    raise ValueError
                return value
        except BaseException:
            raise PlanAError("PRIVATE_TRANSPORT_FAILED") from None

    def get_config(self):
        return self._call("GET")

    def post_config(self, value):
        self._call("POST", value)

    def write_secret(self, successor):
        NativeSecretWriter().write_secret(successor)


class NativeSecretWriter:
    """Private SA-02-owned stdin write only. No deploy/redeploy/upload command."""
    def write_secret(self, successor):
        vector = ["railway", "variable", "set", "API_KEY", "--stdin", "--skip-deploys",
                  "--service", TARGET["service"], "--environment", TARGET["environment"], "--json"]
        try:
            # Existing native auth is opaque. No auth store or variable reads.
            result = subprocess.run(vector, input=successor.encode("ascii"), shell=False,
                                    capture_output=True, timeout=30, check=False)
            if result.returncode != 0:
                raise ValueError
            # stdout/stderr are deliberately private and never in a receipt.
        except BaseException:
            raise PlanAError("PRIVATE_WRITE_FAILED") from None


class ActivationOwnerRequired:
    def preflight(self, req):
        raise PlanAError("ACTIVATION_OWNER_REQUIRED")


class WindowsCustody:
    """CurrentUser DPAPI; existing private owner/DACL required, never repaired.

    Same account/machine only. No memory-zeroization or restore claim. The fixed
    directory must already exist. Reject reparse points in every ancestor. A
    retained lock file is advisory-locked, not deleted on release/failure.
    """
    def __init__(self, operation_id):
        if os.name != "nt" or not SAFE_ID.fullmatch(operation_id):
            raise PlanAError("CUSTODY_UNAVAILABLE")
        local = os.environ.get("LOCALAPPDATA")
        if not local or not Path(local).is_absolute():
            raise PlanAError("CUSTODY_UNAVAILABLE")
        self.directory = Path(local) / "DC" / "PlanA" / "MCPO"
        self.path = self.directory / (operation_id + ".dpapi")
        self._locked = False

    @staticmethod
    def _apis():
        if os.name != "nt":
            raise PlanAError("CUSTODY_UNAVAILABLE")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        adv.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        adv.GetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        adv.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
        adv.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int]
        adv.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        return kernel, adv

    def _qualify(self, path):
        kernel, adv = self._apis()
        # lstat rather than resolve: never traverse a junction/symlink silently.
        for parent in [path] + list(path.parents):
            st = parent.lstat()
            if getattr(st, "st_file_attributes", 0) & 0x400:
                raise PlanAError("CUSTODY_UNAVAILABLE")
        token, size = wintypes.HANDLE(), wintypes.DWORD()
        sd, owner, dacl = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        trusted = []
        try:
            if not adv.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
                raise ValueError
            adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value)
            if not adv.GetTokenInformation(token, 1, buf, size, ctypes.byref(size)):
                raise ValueError
            user_sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
            if adv.GetNamedSecurityInfoW(str(path), 1, 5, ctypes.byref(owner), None,
                                         ctypes.byref(dacl), None, ctypes.byref(sd)) != 0:
                raise ValueError
            if not owner.value or not dacl.value or not adv.EqualSid(owner, user_sid):
                raise ValueError
            # SYSTEM and Administrators are normal Windows custodians. Any
            # other allow trustee (including Users/Everyone) fails closed.
            for sid_text in ("S-1-5-18", "S-1-5-32-544"):
                sid = ctypes.c_void_p()
                if not adv.ConvertStringSidToSidW(sid_text, ctypes.byref(sid)):
                    raise ValueError
                trusted.append(sid)
            class AclSize(ctypes.Structure):
                _fields_ = [("count", wintypes.DWORD), ("used", wintypes.DWORD), ("free", wintypes.DWORD)]
            info = AclSize()
            if not adv.GetAclInformation(dacl, ctypes.byref(info), ctypes.sizeof(info), 2) or info.count == 0:
                raise ValueError
            for i in range(info.count):
                ace = ctypes.c_void_p()
                if not adv.GetAce(dacl, i, ctypes.byref(ace)):
                    raise ValueError
                header = ctypes.string_at(ace, 4)
                kind, flags = header[0], header[1]
                if kind == 1:  # A deny ACE cannot grant access.
                    continue
                if kind != 0:  # No guessing object/callback ACE layouts.
                    raise ValueError
                sid = ctypes.c_void_p(ace.value + 8)
                if not any(adv.EqualSid(sid, candidate) for candidate in [user_sid] + trusted):
                    raise ValueError
                if path.is_dir() and flags & 0x08:  # inherit-only is okay but not own access proof
                    continue
        except BaseException:
            raise PlanAError("CUSTODY_UNAVAILABLE") from None
        finally:
            for sid in trusted:
                kernel.LocalFree(sid)
            if sd.value:
                kernel.LocalFree(sd)
            if token.value:
                kernel.CloseHandle(token)

    @staticmethod
    def _crypt(data, decrypt=False):
        if len(data) > MAX_BLOB or os.name != "nt":
            raise PlanAError("CUSTODY_UNAVAILABLE")
        class Blob(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]
        buffer = ctypes.create_string_buffer(data)
        source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
        output = Blob()
        crypt = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
        fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        try:
            # CRYPTPROTECT_UI_FORBIDDEN only: no LOCAL_MACHINE flag.
            if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
                raise ValueError
            if not 0 < output.size <= MAX_BLOB:
                raise ValueError
            return ctypes.string_at(output.data, output.size)
        except BaseException:
            raise PlanAError("CUSTODY_UNAVAILABLE") from None
        finally:
            if output.data:
                kernel.LocalFree(output.data)

    @contextlib.contextmanager
    def locked(self):
        import msvcrt
        self._qualify(self.directory)
        lock = self.path.with_suffix(".lock")
        if os.path.lexists(lock):
            self._qualify(lock)
        handle = None
        try:
            handle = open(lock, "a+b")
            self._qualify(lock)
            if handle.seek(0, 2) == 0:
                handle.write(b"0"); handle.flush(); os.fsync(handle.fileno())
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            self._locked = True
        except BaseException:
            if handle:
                handle.close()
            raise PlanAError("CUSTODY_UNAVAILABLE") from None
        try:
            yield self
        finally:
            if handle:
                try:
                    if self._locked:
                        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                finally:
                    handle.close()
            self._locked = False

    def ready(self):
        if not self._locked:
            raise PlanAError("CUSTODY_UNAVAILABLE")
        self._qualify(self.directory)
        sentinel = b"PLAN_A_FIXED_FAKE_READINESS_SENTINEL"
        if self._crypt(self._crypt(sentinel), decrypt=True) != sentinel:
            raise PlanAError("CUSTODY_UNAVAILABLE")
        if os.path.lexists(self.path.with_suffix(".pending")):
            raise PlanAError("CUSTODY_UNAVAILABLE")

    def load(self):
        if not self._locked:
            raise PlanAError("CUSTODY_UNAVAILABLE")
        if os.path.lexists(self.path.with_suffix(".pending")):
            raise PlanAError("CUSTODY_UNAVAILABLE")
        if not os.path.lexists(self.path):
            return None
        self._qualify(self.path)
        with open(self.path, "rb") as stream:
            blob = stream.read(MAX_BLOB + 1)
        if len(blob) > MAX_BLOB:
            raise PlanAError("CUSTODY_UNAVAILABLE")
        return json.loads(self._crypt(blob, decrypt=True).decode("utf-8"), object_pairs_hook=_json_pairs)

    def save(self, state):
        if not self._locked or set(state) - {"binding", "successor", "phase", "activation"}:
            raise PlanAError("CUSTODY_UNAVAILABLE")
        self._qualify(self.directory)
        if os.path.lexists(self.path):
            self._qualify(self.path)
            existing = self.load()
            if existing["binding"] != state["binding"] or existing["successor"] != state["successor"]:
                raise PlanAError("CUSTODY_UNAVAILABLE")
        blob = self._crypt(json.dumps(state, separators=(",", ":")).encode("utf-8"))
        pending = self.path.with_suffix(".pending")
        # O_EXCL leaves an uncertain/failed transaction retained, never cleaned up.
        with open(pending, "xb") as stream:
            self._qualify(pending)
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        self._qualify(pending)
        os.replace(pending, self.path)
        self._qualify(self.path)


def private_jwt():
    # Windows getpass uses msvcrt; disallow redirected stdin and fallback echo.
    if os.name != "nt" or not sys.stdin.isatty() or not sys.stderr.isatty():
        raise PlanAError("NO_ECHO_REQUIRED")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass("Existing current Open WebUI admin JWT (private, no echo): ")
        if not value or any(c.isspace() for c in value):
            raise ValueError
        return value
    except BaseException:
        raise PlanAError("NO_ECHO_REQUIRED") from None


class ClosedParser(argparse.ArgumentParser):
    def error(self, message):
        raise PlanAError("INVALID_REQUEST")


def main(argv=None):
    try:
        parser = ClosedParser(description="Private Plan A fixed-action preflight; activation bridge pending.")
        parser.add_argument("action", choices=["rotate-mcpo"])
        for name in ("operation-id", "approval-reference", "project", "environment", "service",
                     "expected-source", "expected-deployment", "qualification-reference"):
            parser.add_argument("--" + name, required=True)
        parser.add_argument("--selector", nargs=3, action="append", required=True, metavar=("URL", "PATH", "INFO_ID"))
        parser.add_argument("--phase", choices=["execute", "resume"], required=True)
        for name in ("logging-qualified", "patched-source-qualified", "exclusive-writer", "auth-available"):
            parser.add_argument("--" + name, action="store_true")
        args = parser.parse_args(argv)
        req = {"operation_id": args.operation_id, "approval_reference": args.approval_reference,
               "expected_target": {"project": args.project, "environment": args.environment, "service": args.service},
               "expected_source": args.expected_source, "expected_deployment": args.expected_deployment,
               "selectors": [dict(zip(("url", "path", "info_id"), values)) for values in args.selector],
               "phase": args.phase, "prerequisites": {
                   "logging_qualified": args.logging_qualified, "patched_source_qualified": args.patched_source_qualified,
                   "exclusive_writer": args.exclusive_writer, "auth_available": args.auth_available,
                   "qualification_reference": args.qualification_reference}}
        validate_request(req)
        # Hardcoded fail-closed bridge: confirmations are not native evidence.
        # No key, JWT prompt, DPAPI call, directory/lock creation or GET occurs.
        ActivationOwnerRequired().preflight(req)
        # Private bootstrap building blocks above intentionally remain unconnected
        # until exact native activation ownership/method is qualified and approved.
    except PlanAError as exc:
        result = receipt("REJECTED", exc.code)
    except BaseException:
        result = receipt("REJECTED", "INTERNAL_STOP")
    else:
        result = receipt("REJECTED", "ACTIVATION_OWNER_REQUIRED")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
