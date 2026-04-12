import os
import time
import httpx

OPENWEBUI_URL = os.environ.get("OPENWEBUI_INTERNAL_URL", "http://openwebui:8080")
ADMIN_EMAIL = os.environ.get("WEBUI_ADMIN_EMAIL", "admin@gpthub.local")
ADMIN_PASSWORD = os.environ.get("WEBUI_ADMIN_PASSWORD", "change_me_admin")
ADMIN_NAME = os.environ.get("WEBUI_ADMIN_NAME", "Admin")
TOOLS_DIR = "/tools"


def wait_for_openwebui(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{OPENWEBUI_URL}/ready", timeout=5)
            if r.status_code == 200:
                print("OpenWebUI ready")
                return
        except Exception:
            pass
        print("Waiting for OpenWebUI...")
        time.sleep(4)
    raise RuntimeError("OpenWebUI did not become ready")


def get_token() -> str:
    r = httpx.post(
        f"{OPENWEBUI_URL}/api/v1/auths/signin",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        timeout=10,
    )
    if r.status_code == 200:
        return r.json()["token"]

    if r.status_code in (400, 401, 404):
        print("Admin not found, attempting signup...")
        sr = httpx.post(
            f"{OPENWEBUI_URL}/api/v1/auths/signup",
            json={"name": ADMIN_NAME, "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
            timeout=10,
        )
        if sr.status_code == 200:
            print("Admin account created")
            r2 = httpx.post(
                f"{OPENWEBUI_URL}/api/v1/auths/signin",
                json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
                timeout=10,
            )
            if r2.status_code == 200:
                return r2.json()["token"]
        print(f"Signup response: {sr.status_code} {sr.text[:200]}")

    raise RuntimeError(f"Auth failed: {r.status_code}")


def list_existing_tools(token: str) -> set[str]:
    r = httpx.get(
        f"{OPENWEBUI_URL}/api/v1/tools/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if r.status_code != 200:
        return set()
    return {t["id"] for t in r.json()}


def register_tool(token: str, tool_id: str, name: str, content: str) -> None:
    r = httpx.post(
        f"{OPENWEBUI_URL}/api/v1/tools/create",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"id": tool_id, "name": name, "content": content, "meta": {"description": name}},
        timeout=15,
    )
    if r.status_code in (200, 201):
        print(f"Registered: {name}")
    elif "already" in r.text.lower():
        print(f"Already exists: {name}")
    else:
        print(f"Failed {name}: {r.status_code} {r.text[:200]}")


def main() -> None:
    wait_for_openwebui()
    time.sleep(3)

    try:
        token = get_token()
    except RuntimeError as e:
        print(f"Auth error: {e}")
        raise

    existing = list_existing_tools(token)

    tools = [
        ("memory_viewer", "GPTHub Memory Viewer", "memory_viewer.py"),
        ("workspaces", "GPTHub Workspaces", "workspaces.py"),
    ]

    for tool_id, name, filename in tools:
        if tool_id in existing:
            print(f"Already registered: {name}")
            continue
        path = os.path.join(TOOLS_DIR, filename)
        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue
        with open(path) as f:
            content = f.read()
        register_tool(token, tool_id, name, content)

    print("Init complete")


if __name__ == "__main__":
    main()
