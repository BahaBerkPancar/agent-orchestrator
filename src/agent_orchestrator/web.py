"""Small dependency-free localhost approval UI."""

from __future__ import annotations

import hmac
import html
import json
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit

from .engine import Engine, RunBusy, UsageError

MAX_SESSIONS = 1000
MAX_FORM_BODY = 65536

STYLE = """
:root { color-scheme: light dark; font: 16px/1.45 system-ui, sans-serif; }
body { max-width: 70rem; margin: 2rem auto; padding: 0 1rem; }
a { color: #2878d0; } table { width: 100%; border-collapse: collapse; }
th, td { padding: .55rem; text-align: left; border-bottom: 1px solid #8886; }
code, pre, textarea { font-family: ui-monospace, monospace; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #8881; padding: .8rem; }
.pending { border: 2px solid #d68b00; border-radius: .5rem; padding: 1rem; }
form { display: inline-block; margin: .25rem .4rem .25rem 0; vertical-align: top; }
button, textarea { font: inherit; padding: .45rem; } textarea { display: block; width: min(40rem, 85vw); }
.error { color: #c32; font-weight: 650; }
"""


@dataclass
class WebResponse:
    status: int
    body: bytes
    headers: dict


class WebApplication:
    def __init__(self, home, shop_db=None, port=8070):
        self.home = home
        self.shop_db = shop_db
        self.port = port
        self.sessions = OrderedDict()

    @staticmethod
    def _header(headers, name):
        return next(
            (value for key, value in headers.items() if key.lower() == name.lower()),
            None,
        )

    def host_allowed(self, headers):
        host = self._header(headers, "Host")
        return host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def _engine(self):
        return Engine(self.home, shop_db=self.shop_db)

    def _session(self, headers):
        cookie = SimpleCookie(self._header(headers, "Cookie") or "")
        morsel = cookie.get("agent_session")
        if morsel and morsel.value in self.sessions:
            self.sessions.move_to_end(morsel.value)
            return morsel.value, None
        token = secrets.token_urlsafe(32)
        self.sessions[token] = None
        if len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)
        return token, f"agent_session={token}; HttpOnly; SameSite=Strict; Path=/"

    @staticmethod
    def _page(title, content):
        safe_title = html.escape(title)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{safe_title}</title><style>{STYLE}</style></head>"
            f"<body><header><a href='/'>agent-orchestrator</a></header><h1>{safe_title}</h1>"
            f"{content}</body></html>"
        ).encode()

    @staticmethod
    def _status_text(status):
        return {
            200: "OK",
            303: "See Other",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
            409: "Conflict",
            421: "Misdirected Request",
        }.get(status, "Error")

    def _response(self, status, content, cookie=None, location=None):
        headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        }
        if cookie:
            headers["Set-Cookie"] = cookie
        if location:
            headers["Location"] = location
        return WebResponse(status, content, headers)

    def _list(self, engine):
        rows = []
        for run_id in reversed(engine.store.run_ids()):
            state = engine.store.load(run_id)
            rows.append(
                "<tr>"
                f"<td><a href='/runs/{quote(run_id, safe='')}'>{html.escape(run_id)}</a></td>"
                f"<td>{html.escape(state.status)}</td>"
                f"<td>{html.escape(state.role)}</td>"
                f"<td>{html.escape(state.goal)}</td>"
                "</tr>"
            )
        content = (
            "<table><thead><tr><th>Run</th><th>Status</th><th>Role</th><th>Goal</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            if rows
            else "<p>No runs yet.</p>"
        )
        return self._page("Runs", content)

    @staticmethod
    def _pending(engine, state):
        while state.active_child:
            state = engine.store.load(state.active_child)
        return state if state.status == "waiting_approval" and state.pending else None

    def _detail(self, engine, run_id, csrf, error=None):
        state = engine.store.load(run_id)
        pending = self._pending(engine, state)
        sections = [
            f"<p>Status: <strong>{html.escape(state.status)}</strong></p>",
            f"<p>Goal: {html.escape(state.goal)}</p>",
        ]
        if error:
            sections.append(f"<p class='error'>{html.escape(error)}</p>")
        if pending:
            args = json.dumps(pending.pending["args"], indent=2, sort_keys=True)
            common = f"<input type='hidden' name='csrf' value='{html.escape(csrf)}'>"
            sections.append(
                "<section class='pending'><h2>Pending approval</h2>"
                f"<p>Tool: <code>{html.escape(pending.pending['name'])}</code></p>"
                f"<p>Risk: <strong>{html.escape(pending.pending['risk'])}</strong></p>"
                f"<pre>{html.escape(args)}</pre>"
                f"<form method='post' action='/runs/{quote(run_id, safe='')}/decision'>{common}"
                "<input type='hidden' name='decision' value='approve'><button>Approve</button></form>"
                f"<form method='post' action='/runs/{quote(run_id, safe='')}/decision'>{common}"
                "<input type='hidden' name='decision' value='reject'><button>Reject</button></form>"
                f"<form method='post' action='/runs/{quote(run_id, safe='')}/decision'>{common}"
                "<input type='hidden' name='decision' value='edit'><label>Edit arguments as JSON"
                f"<textarea name='args' rows='9'>{html.escape(args)}</textarea></label>"
                "<button>Validate and approve edit</button></form></section>"
            )
        events = []
        for event in engine.store.events(run_id):
            payload = html.escape(json.dumps(event["data"], sort_keys=True))
            events.append(
                f"<tr><td>{event['seq']}</td><td>{html.escape(event['type'])}</td>"
                f"<td><code>{payload}</code></td></tr>"
            )
        sections.append(
            "<h2>Event timeline</h2><table><thead><tr><th>#</th><th>Type</th><th>Data</th></tr></thead>"
            f"<tbody>{''.join(events)}</tbody></table>"
        )
        return self._page(f"Run {run_id}", "".join(sections))

    def handle(self, method, target, headers=None, body=b""):
        headers = headers or {}
        if not self.host_allowed(headers):
            return self._response(
                421,
                self._page("Misdirected request", "<p>Invalid Host header.</p>"),
            )
        path = urlsplit(target).path
        session, cookie = self._session(headers)
        engine = self._engine()
        try:
            if method == "GET" and path == "/":
                return self._response(200, self._list(engine), cookie)
            parts = [part for part in path.split("/") if part]
            if method == "GET" and len(parts) == 2 and parts[0] == "runs":
                try:
                    content = self._detail(engine, parts[1], session)
                except KeyError:
                    content = self._page("Not found", "<p>Unknown run.</p>")
                    return self._response(404, content, cookie)
                return self._response(200, content, cookie)
            if (
                method == "POST"
                and len(parts) == 3
                and parts[0] == "runs"
                and parts[2] == "decision"
            ):
                if len(body) > MAX_FORM_BODY:
                    raise UsageError("form body is too large")
                form = parse_qs(body.decode("utf-8", errors="strict"), keep_blank_values=True)
                supplied = form.get("csrf", [""])[0]
                if not supplied or not hmac.compare_digest(supplied, session):
                    return self._response(
                        403,
                        self._page("Forbidden", "<p>CSRF token rejected.</p>"),
                        cookie,
                    )
                decision = form.get("decision", [""])[0]
                if decision not in {"approve", "reject", "edit"}:
                    raise UsageError("unknown approval decision")
                args = None
                if decision == "edit":
                    try:
                        args = json.loads(form.get("args", [""])[0])
                    except json.JSONDecodeError as exc:
                        raise UsageError(f"edited args are not valid JSON: {exc}") from None
                reason = {
                    "approve": "web approval",
                    "edit": "web edit",
                    "reject": "web rejection",
                }[decision]
                engine.decide(parts[1], decision, args, reason)
                return self._response(
                    303,
                    b"",
                    cookie,
                    location=f"/runs/{quote(parts[1], safe='')}",
                )
            return self._response(
                404, self._page("Not found", "<p>Unknown page.</p>"), cookie
            )
        except KeyError:
            return self._response(
                404, self._page("Not found", "<p>Unknown run.</p>"), cookie
            )
        except UnicodeDecodeError:
            return self._response(
                400, self._page("Bad request", "<p>Form must be UTF-8.</p>"), cookie
            )
        except UsageError as exc:
            content = self._page("Bad request", f"<p>{html.escape(str(exc))}</p>")
            return self._response(400, content, cookie)
        except RunBusy as exc:
            content = self._page("Run busy", f"<p>{html.escape(str(exc))}</p>")
            return self._response(409, content, cookie)
        finally:
            engine.store.close()


def parse_content_length(value):
    if value in (None, ""):
        return 0
    try:
        length = int(value)
    except (TypeError, ValueError):
        raise UsageError("Content-Length must be a non-negative integer") from None
    if length < 0:
        raise UsageError("Content-Length must be a non-negative integer")
    if length > MAX_FORM_BODY:
        raise UsageError("form body is too large")
    return length


def serve(home, shop_db=None, port=8070):
    app = WebApplication(home, shop_db, port)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, response):
            self.send_response(response.status, app._status_text(response.status))
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

        def _handle(self):
            headers = dict(self.headers)
            if not app.host_allowed(headers):
                self._send(app.handle(self.command, self.path, headers))
                return
            try:
                length = parse_content_length(app._header(headers, "Content-Length"))
            except UsageError as exc:
                self._send(
                    app._response(
                        400,
                        app._page("Bad request", f"<p>{html.escape(str(exc))}</p>"),
                    )
                )
                return
            response = app.handle(
                self.command,
                self.path,
                headers,
                self.rfile.read(length) if length else b"",
            )
            self._send(response)

        do_GET = _handle
        do_POST = _handle

        def log_message(self, format_, *args):
            print(f"web: {format_ % args}")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"approval UI: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
