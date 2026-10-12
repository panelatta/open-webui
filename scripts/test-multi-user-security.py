"""HTTP isolation regression suite; run only in the documented network-isolated test container."""

import io
import json
import os
import pathlib
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyotp
import requests
import socketio

assert os.environ.get("WEBUI_SECRET_KEY") == "security-tests-only-not-production-secret"
assert os.environ.get("DATA_DIR", "").startswith("/tmp/security-")
BASE = "http://127.0.0.1:8080"
DB = pathlib.Path(os.environ["DATA_DIR"]) / "webui.db"
OUT = pathlib.Path("/audit/http-security-results.json")
password = "Synthetic-Account-Passphrase-2026!"
results, upstream_calls, uploads, responses = [], [], {}, {}


def check(name, condition, **detail):
    results.append({"name": name, "passed": bool(condition), **detail})
    OUT.write_text(json.dumps(results, indent=2))
    assert condition, (name, detail)
    print(name, "PASS", flush=True)


def call(method, path, token=None, **kw):
    headers = kw.pop("headers", {})
    if token:
        headers["Authorization"] = "Bearer " + token
    return requests.request(method, BASE + path, headers=headers, timeout=30, **kw)


def ok(method, path, token=None, **kw):
    r = call(method, path, token, **kw)
    assert r.status_code == 200, (path, r.status_code, r.text[:300])
    return r.json()


def config(values):
    with sqlite3.connect(DB) as db:
        for key, value in values.items():
            db.execute(
                "insert into config(key,value,updated_at) values(?,?,?) on conflict(key) do update set value=excluded.value",
                (key, json.dumps(value), int(time.time())),
            )


def response(text):
    rid = "resp-" + uuid.uuid4().hex
    obj = {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "audit-public",
        "output": [
            {
                "id": "msg-" + uuid.uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    responses[rid] = obj
    return obj


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, obj, status=200):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        upstream_calls.append(self.path)
        if self.path.endswith("/models"):
            return self.send(
                {
                    "data": [
                        {"id": "audit-public", "object": "model", "owned_by": "audit"}
                    ]
                }
            )
        rid = self.path.split("/responses/")[-1].split("?")[0]
        self.send(
            responses[rid] if rid in responses else {"error": "unknown"},
            200 if rid in responses else 404,
        )

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        upstream_calls.append(self.path)
        if self.path.endswith("/files"):
            fid = "file-" + uuid.uuid4().hex
            uploads[fid] = (
                "OWNER_A_FILE_SENTINEL"
                if b"OWNER_A_FILE_SENTINEL" in raw
                else "OWN_FILE"
            )
            return self.send({"id": fid, "object": "file"})
        data = json.loads(raw)
        text = "MODEL_OK"
        serialized = json.dumps(data)
        for fid, content in uploads.items():
            if fid in serialized:
                text = content
        if data.get("previous_response_id") in responses:
            text = responses[data["previous_response_id"]]["output"][0]["content"][0][
                "text"
            ]
        elif isinstance(data.get("input"), str):
            text = data["input"]
        obj = response(text)
        if data.get("stream"):
            stream = "".join(
                "data: " + json.dumps({"type": kind, "response": obj}) + "\n\n"
                for kind in ["response.created", "response.completed"]
            )
            raw = stream.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send(obj)


for attempt in range(30):
    try:
        if call("GET", "/ready").status_code == 200:
            break
    except requests.ConnectionError:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Isolated test server is not ready")
server = ThreadingHTTPServer(("127.0.0.1", 8081), Provider)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    admin = ok(
        "POST",
        "/api/v1/auths/signup",
        json={
            "name": "Admin",
            "email": "admin@audit.example.com",
            "password": password,
        },
    )
    adm = admin["token"]
    config(
        {
            "ui.enable_signup": True,
            "ui.default_user_role": "pending",
            "auth.jwt_expiry": "24h",
            "auth.admin.show": False,
        }
    )
    r = call(
        "POST",
        "/api/v1/auths/signup",
        json={"name": "Weak", "email": "weak@audit.example.com", "password": ""},
    )
    check("empty_password_signup_rejected", r.status_code == 400, status=r.status_code)
    r = call(
        "POST",
        "/api/v1/auths/signup",
        json={"name": "Weak", "email": "short@audit.example.com", "password": "short"},
    )
    check("short_password_signup_rejected", r.status_code == 400, status=r.status_code)
    pending = ok(
        "POST",
        "/api/v1/auths/signup",
        json={
            "name": "Pending",
            "email": "pending@audit.example.com",
            "password": password,
            "role": "admin",
        },
    )
    check("signup_cannot_choose_admin_role", pending["role"] == "pending")
    for label, token in [("anonymous", None), ("pending", pending["token"])]:
        for path in [
            "/api/v1/chats/",
            "/api/v1/models/list",
            "/api/v1/files/",
            "/api/v1/memories/",
            "/api/v1/auths/admin/details",
        ]:
            r = call("GET", path, token)
            check(label + "_" + path, r.status_code in (400, 401, 403))
    users = {
        x: ok(
            "POST",
            "/api/v1/auths/add",
            adm,
            json={
                "name": x,
                "email": x + "@audit.example.com",
                "password": password,
                "role": "user",
            },
        )
        for x in ("a", "b", "session")
    }
    a, b, s = [users[x]["token"] for x in ("a", "b", "session")]
    r = call(
        "GET", "/api/v1/auths/", b, headers={"Origin": "https://untrusted.example"}
    )
    check(
        "untrusted_origin_has_no_cors_permission",
        not r.headers.get("Access-Control-Allow-Origin"),
    )
    r = call(
        "GET", "/api/v1/auths/", b, headers={"Origin": "https://openwebui.panelatta.de"}
    )
    check(
        "trusted_origin_allowed",
        r.headers.get("Access-Control-Allow-Origin")
        == "https://openwebui.panelatta.de",
    )
    check(
        "auth_cookie_secure_httponly",
        "; Secure" in r.headers.get("Set-Cookie", "")
        and "HttpOnly" in r.headers.get("Set-Cookie", ""),
    )
    check(
        "signout_success", call("POST", "/api/v1/auths/signout", s).status_code == 200
    )
    check(
        "signed_out_token_rejected", call("GET", "/api/v1/chats/", s).status_code == 401
    )
    check(
        "other_session_unaffected", call("GET", "/api/v1/chats/", b).status_code == 200
    )
    s = ok(
        "POST",
        "/api/v1/auths/signin",
        json={"email": "session@audit.example.com", "password": password},
    )["token"]
    check(
        "password_change_success",
        call(
            "POST",
            "/api/v1/auths/update/password",
            s,
            json={"password": password, "new_password": password + "New"},
        ).status_code
        == 200,
    )
    check(
        "password_change_revokes_session",
        call("GET", "/api/v1/chats/", s).status_code == 401,
    )
    config(
        {
            "openai.enable": True,
            "openai.api_base_urls": [
                "http://127.0.0.1:8081/v1",
                "http://127.0.0.1:8082/v1",
            ],
            "openai.api_keys": ["test", "other-test"],
            "openai.api_configs": {
                "0": {
                    "enable": True,
                    "api_type": "responses",
                    "model_ids": ["audit-public"],
                },
                "1": {"enable": False, "model_ids": []},
            },
        }
    )
    grant = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
    public = {
        "id": "audit-public",
        "name": "Audit Public",
        "meta": {},
        "params": {"system": "PRIVATE_SYSTEM_SENTINEL"},
        "access_grants": grant,
    }
    ok("POST", "/api/v1/models/create", adm, json=public)
    ok(
        "POST",
        "/api/v1/models/create",
        adm,
        json={
            **public,
            "id": "audit-private",
            "base_model_id": "audit-public",
            "access_grants": [],
        },
    )
    with sqlite3.connect(DB) as db:
        db.execute(
            "update model set user_id=? where id=?", (users["a"]["id"], "audit-private")
        )
    for token, label, expected in [
        (a, "owner", True),
        (b, "other", False),
        (adm, "admin", True),
        (b, "other_after_admin", False),
    ]:
        r = call("GET", "/api/models", token)
        check("model_cache_" + label, ("audit-private" in r.text) == expected)
    for path in [
        "/api/v1/models/model?id=audit-private",
        "/api/v1/models/model/history?id=audit-private",
    ]:
        check("private_model_" + path, call("GET", path, b).status_code in (401, 403))
    r = call("GET", "/api/v1/models/model?id=audit-public", b)
    check(
        "shared_model_hides_system_prompt",
        r.status_code == 200 and "PRIVATE_SYSTEM_SENTINEL" not in r.text,
    )
    msg = {
        "id": "m-a",
        "role": "user",
        "content": "OWNER_A_CHAT_SENTINEL",
        "parentId": None,
        "childrenIds": [],
    }
    chat = ok(
        "POST",
        "/api/v1/chats/new",
        a,
        json={
            "chat": {
                "title": "Private",
                "history": {"currentId": "m-a", "messages": {"m-a": msg}},
                "messages": [msg],
            }
        },
    )
    cid = chat["id"]
    file = ok(
        "POST",
        "/api/v1/files/?process=false",
        a,
        files={"file": ("private.txt", b"OWNER_A_FILE_SENTINEL", "text/plain")},
    )
    fid = file["id"]
    for path in [
        "/api/v1/chats/",
        "/api/v1/chats/manage",
        "/api/v1/chats/search?text=OWNER_A",
        "/api/v1/chats/all",
        "/api/v1/files/",
        "/api/v1/memories/",
    ]:
        r = call("GET", path, b)
        check(
            "list_isolation_" + path,
            r.status_code == 200 and "OWNER_A_" not in r.text and fid not in r.text,
        )
    for method, path, kw in [
        ("GET", "/api/v1/chats/" + cid, {}),
        ("POST", "/api/v1/chats/" + cid, {"json": {"chat": {"title": "tamper"}}}),
        ("POST", "/api/v1/chats/" + cid + "/clone", {"json": {}}),
        ("POST", "/api/v1/chats/" + cid + "/fork", {"json": {}}),
        ("DELETE", "/api/v1/chats/" + cid, {}),
        ("GET", "/api/v1/files/" + fid + "/content", {}),
        ("DELETE", "/api/v1/files/" + fid, {}),
    ]:
        r = call(method, path, b, **kw)
        check(
            "object_isolation_" + method + "_" + path, r.status_code in (401, 403, 404)
        )
    r = call("POST", "/api/v1/chats/bulk", b, json={"ids": [cid], "action": "delete"})
    check(
        "bulk_foreign_chat_rejected",
        r.status_code == 200 and r.json()[0]["success"] is False,
    )
    chat_payload = {
        "model": "audit-public",
        "messages": [{"role": "user", "content": "Read attachment"}],
        "stream": False,
    }
    attachment = {"type": "file", "id": fid, "name": "private.txt"}
    for path, extra in [
        ("/api/chat/completions", {"files": [attachment]}),
        ("/openai/chat/completions", {"files": [attachment]}),
        ("/openai/chat/completions", {"metadata": {"files": [attachment]}}),
    ]:
        before = len(uploads)
        r = call("POST", path, b, json={**chat_payload, **extra})
        check(
            "foreign_hosted_attachment_" + path + "_" + next(iter(extra)),
            r.status_code in (400, 401, 403) and len(uploads) == before,
            status=r.status_code,
        )
    r = call(
        "POST", "/api/chat/completions", a, json={**chat_payload, "files": [attachment]}
    )
    check(
        "owner_hosted_attachment_works",
        r.status_code == 200 and "OWNER_A_FILE_SENTINEL" in r.text,
        status=r.status_code,
    )
    fdata = ok("GET", "/api/v1/files/" + fid, a)["data"]
    up_id = fdata["openai_file_id"]
    for payload in [
        {"previous_response_id": "unknown-response", "input": "read"},
        {
            "input": [
                {"role": "user", "content": [{"type": "input_file", "file_id": up_id}]}
            ]
        },
        {"conversation": "unknown-conversation"},
        {"input": [{"type": "item_reference", "id": "unknown-item"}]},
    ]:
        before = len(upstream_calls)
        r = call(
            "POST", "/openai/responses", b, json={"model": "audit-public", **payload}
        )
        check(
            "foreign_native_reference_" + str(payload)[:75],
            r.status_code == 403 and len(upstream_calls) == before,
        )
    r = call(
        "POST",
        "/openai/responses",
        a,
        json={
            "model": "audit-public",
            "input": [
                {"role": "user", "content": [{"type": "input_file", "file_id": up_id}]}
            ],
        },
    )
    check(
        "owner_native_file_reference",
        r.status_code == 200 and "OWNER_A_FILE_SENTINEL" in r.text,
    )
    r = call(
        "POST",
        "/openai/responses",
        a,
        json={
            "model": "audit-public",
            "input": "OWNER_A_RESPONSE_SENTINEL",
            "stream": True,
        },
    )
    events = [
        json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")
    ]
    rid = events[0]["response"]["id"]
    check(
        "owner_stream_still_complete",
        r.status_code == 200 and events[-1]["type"] == "response.completed",
    )
    for token, label, wanted in [(b, "foreign", 403), (a, "owner", 200)]:
        r = call(
            "POST",
            "/openai/responses",
            token,
            json={
                "model": "audit-public",
                "previous_response_id": rid,
                "input": "repeat",
            },
        )
        check("previous_response_" + label, r.status_code == wanted)
        r = call("GET", "/openai/responses/" + rid + "?model=audit-public", token)
        check("background_retrieve_" + label, r.status_code == wanted)
    r = call("GET", "/openai/responses/" + rid + "?model=audit-public&route_idx=1", a)
    check("disabled_route_cannot_resume", r.status_code == 403)
    shared = {
        **public,
        "id": "shared-file",
        "base_model_id": "audit-public",
        "meta": {"knowledge": [{"id": fid, "type": "file", "name": "private.txt"}]},
    }
    ok("POST", "/api/v1/models/create", adm, json=shared)
    r = call(
        "POST",
        "/openai/chat/completions",
        b,
        json={**chat_payload, "files": [attachment]},
    )
    check(
        "explicit_shared_file_still_works",
        r.status_code == 200 and "OWNER_A_FILE_SENTINEL" in r.text,
    )
    with sqlite3.connect(DB) as db:
        db.execute(
            'delete from access_grant where resource_type="model" and resource_id="shared-file"'
        )
    r = call(
        "POST",
        "/openai/chat/completions",
        b,
        json={**chat_payload, "files": [attachment]},
    )
    check("revoked_file_share_blocks_reuse", r.status_code == 403)
    share = ok("POST", "/api/v1/chats/" + cid + "/share", a, json={})
    sid = share["share_id"]
    check(
        "private_share_read_denied",
        call("GET", "/api/v1/chats/share/" + sid, b).status_code == 401,
    )
    check(
        "private_share_clone_denied",
        call("POST", "/api/v1/chats/" + sid + "/clone/shared", b).status_code == 401,
    )
    for token, label, wanted in [(b, "foreign", False), (a, "owner", True)]:
        sock = socketio.Client(
            reconnection=False,
            websocket_extra_options={"origin": "https://openwebui.panelatta.de"},
        )
        try:
            sock.connect(
                BASE,
                socketio_path="/ws/socket.io",
                auth={"token": token},
                transports=["websocket"],
            )
            result = sock.call(
                "events:chat", {"chat_id": cid, "data": {"type": "join"}}, timeout=5
            )
            check("socket_chat_join_" + label, result is wanted)
        finally:
            sock.disconnect()
    # No embeddings are invoked by this suite.
    for path in ["/api/embeddings", "/openai/embeddings", "/ollama/api/embed"]:
        check(
            "embedding_policy_" + path,
            call(
                "POST", path, a, json={"model": "unused", "input": "unused"}
            ).status_code
            == 403,
        )
    with sqlite3.connect(DB) as db:
        db.execute(
            "insert into memory(id,user_id,type,content,created_at,updated_at) values(?,?,?,?,?,?)",
            ("memory-a", users["a"]["id"], "context", "OWNER_A_MEMORY_SENTINEL", 1, 1),
        )
    r = call(
        "POST",
        "/api/v1/memories/search",
        b,
        json={"query": "OWNER_A", "memory_id": "memory-a"},
    )
    check("memory_search_isolation", "OWNER_A_" not in r.text)
    check(
        "memory_update_isolation",
        call(
            "POST", "/api/v1/memories/memory-a/update", b, json={"content": "tamper"}
        ).status_code
        == 404,
    )
    codes = [
        call(
            "POST",
            "/api/v1/auths/signup",
            json={
                "name": "Rate",
                "email": f"rate{i}@audit.example.com",
                "password": password,
            },
        ).status_code
        for i in range(3)
    ]
    check("signup_rate_limited", 429 in codes, statuses=codes)
    config({"auth.mfa.enable": True})
    check("mfa_blocks_old_session", call("GET", "/api/v1/chats/", b).status_code == 401)
    challenge = ok(
        "POST",
        "/api/v1/auths/signin",
        json={"email": "b@audit.example.com", "password": password},
    )
    check(
        "mfa_password_only_returns_challenge",
        "token" not in challenge and challenge["next_step"] == "enroll",
    )
    token = challenge["challenge_token"]
    check(
        "mfa_challenge_not_a_session",
        call("GET", "/api/v1/chats/", token).status_code == 401,
    )
    setup = ok(
        "POST", "/api/v1/auths/mfa/enroll/start", json={"challenge_token": token}
    )
    code = pyotp.TOTP(setup["manual_key"]).now()
    r = call(
        "POST",
        "/api/v1/auths/mfa/enroll/confirm",
        json={"challenge_token": token, "code": code},
    )
    check("mfa_enrollment_works", r.status_code == 200)
    check(
        "mfa_authenticated_session_works",
        call("GET", "/api/v1/chats/", r.json()["token"]).status_code == 200,
    )
    r = call(
        "POST",
        "/api/v1/auths/mfa/enroll/confirm",
        json={"challenge_token": token, "code": code},
    )
    check("mfa_challenge_replay_rejected", r.status_code == 401)
    pending_login = ok(
        "POST",
        "/api/v1/auths/signin",
        json={"email": "pending@audit.example.com", "password": password},
    )
    check(
        "pending_login_still_shows_approval_state",
        pending_login.get("next_step") == "pending" and "token" not in pending_login,
    )
    print("HTTP SECURITY CHECKS:", len(results), flush=True)
finally:
    server.shutdown()
