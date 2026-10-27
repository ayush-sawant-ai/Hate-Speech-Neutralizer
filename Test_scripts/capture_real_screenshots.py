"""Capture pixel-perfect browser screenshots of the Streamlit workspace using Playwright."""

import subprocess
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url="http://localhost:8501/_stcore/health", timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main():
    print("Launching Streamlit server...")
    server = subprocess.Popen(
        [
            str(BASE_DIR / ".venv" / "bin" / "streamlit"),
            "run",
            "app.py",
            "--server.port=8501",
            "--server.headless=true",
        ],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            print("Server timed out.")
            return

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            scenarios = [
                ("01_empty_state", 1280, 900, "http://localhost:8501"),
                ("02_analysis_state", 1280, 900, "http://localhost:8501/?demo=analyzed"),
                ("03_successful_rewrite", 1280, 900, "http://localhost:8501/?demo=rewritten"),
                ("04_failure_state", 1280, 900, "http://localhost:8501/?demo=failure"),
                ("05_tablet_layout", 768, 1024, "http://localhost:8501/?demo=rewritten"),
                ("06_mobile_layout", 375, 812, "http://localhost:8501/?demo=rewritten"),
            ]

            for name, w, h, url in scenarios:
                context = browser.new_context(viewport={"width": w, "height": h})
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded")
                # Wait for the Streamlit DOM elements to mount
                page.wait_for_selector('textarea', timeout=15000)
                time.sleep(2.0)

                out_path = SCREENSHOT_DIR / f"{name}.png"
                page.screenshot(path=str(out_path), full_page=False)
                print(f"Captured: {out_path.name} ({out_path.stat().st_size} bytes, {w}x{h})")
                context.close()

            browser.close()

    finally:
        server.terminate()
        server.wait()
        print("Streamlit server stopped.")


if __name__ == "__main__":
    main()
