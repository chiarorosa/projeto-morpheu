"""Entry point: `python -m morpheu [--browser|--server] [--port N]`."""

import argparse
import os
import threading
import time
import urllib.request

from .config import PROJECT_ROOT

# Keep downloaded model weights inside the project folder, not the user's global cache.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models"))


def wait_until_up(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"O servidor não respondeu em {url}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="morpheu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--browser", action="store_true", help="abrir no navegador em vez da janela desktop")
    mode.add_argument("--server", action="store_true", help="só a API, sem interface")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MORPHEU_PORT", 8765)))
    args = parser.parse_args()

    import uvicorn

    from .app import create_app

    url = f"http://127.0.0.1:{args.port}"
    # Bind to loopback only: the API has no authentication.
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=args.port, log_level="info"))

    if args.server:
        server.run()
        return

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_until_up(f"{url}/api/v1/health")

    if args.browser:
        import webbrowser

        webbrowser.open(url)
        thread.join()
        return

    import webview

    webview.create_window("Morpheu", url, width=1280, height=860, min_size=(900, 640), background_color="#0f1117")
    webview.start()
    server.should_exit = True
    thread.join(timeout=10)


if __name__ == "__main__":
    main()
