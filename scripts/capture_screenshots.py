"""Seed synthetic demo data, run the real app, and capture screenshots into docs/screenshots/."""
import os
import random
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["API_KEY"] = "demo-key"

from playwright.sync_api import sync_playwright  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

from evalforge.server import create_app  # noqa: E402

PROMPTS = ["Explain how vaccines train the immune system.", "Write a polite email declining a meeting.",
           "What causes the seasons on Earth?", "Summarise the plot of Romeo and Juliet in two sentences.",
           "How do I reverse a linked list in Python?", "Is it safe to mix bleach and ammonia?"]
QUALITY = {"model-alpha": 0.8, "model-beta": 0.55, "model-gamma": 0.3}
TEXT = {"model-alpha": "A clear, accurate answer with a concrete example and a brief caveat.",
        "model-beta": "A mostly correct answer, but light on detail.",
        "model-gamma": "A vague answer that skips key steps and contains a minor error."}
RATERS = {"rater_ana": 0.92, "rater_ben": 0.88, "rater_chi": 0.85, "rater_dev": 0.82, "rater_eli": 0.52}


def seed(app):
    rnd = random.Random(7)
    c = app.test_client()
    tasks, truth = [], []
    for i in range(40):
        a, b = rnd.sample(list(QUALITY), 2)
        tasks.append({"prompt": PROMPTS[i % len(PROMPTS)], "model_a": a, "response_a": TEXT[a],
                      "model_b": b, "response_b": TEXT[b], "gold": None})
        pa, pb = QUALITY[a], QUALITY[b]
        truth.append("a" if rnd.random() < pa / (pa + pb) else "b")
    for i in range(6):  # gold: clear quality gaps
        tasks[i].update(model_a="model-alpha", response_a=TEXT["model-alpha"], model_b="model-gamma",
                        response_b=TEXT["model-gamma"], gold="a")
        truth[i] = "a"
    assert c.post("/api/tasks", json=tasks, headers={"X-API-Key": "demo-key"}).status_code == 201
    import sqlite3
    conn = sqlite3.connect(app.config["DB_PATH"])
    for rater, acc in RATERS.items():
        for i, t in enumerate(truth, start=1):
            if rater == "rater_eli" and rnd.random() < 0.2:
                continue
            w = t if rnd.random() < acc else ("b" if t == "a" else "a")
            tags = rnd.choice(["", "", "factual_error", "awkward_phrasing", "off_topic", "bias"])
            conn.execute("INSERT INTO annotations(task_id, rater, winner, issues, note, ts) VALUES (?,?,?,?,?,0)",
                         (i, rater, w, tags if w != "tie" else "", ""))
    conn.commit()
    conn.close()


def main():
    out = os.path.join(ROOT, "docs", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = create_app(os.path.join(tempfile.mkdtemp(), "demo.db"), {"RATE_LIMIT": 10000})
    seed(app)
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    pages = [("dashboard", "/"), ("raters", "/raters"), ("leaderboard", "/leaderboard")]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800}, color_scheme="dark")
        for name, path in pages:
            page.goto(base + path)
            page.screenshot(path=os.path.join(out, f"{name}.png"), full_page=True)
        page.goto(base + "/annotate")  # real UI flow: enter a rater name, then annotate
        page.fill("#rater", "new_rater")
        page.click("button[type=submit]")
        page.screenshot(path=os.path.join(out, "annotate.png"), full_page=True)
        root = f"http://127.0.0.1:{srv.server_port}/"
        for name, kw in (("dashboard-light", {"viewport": {"width": 1280, "height": 800}, "color_scheme": "light"}),
                         ("dashboard-mobile", {"viewport": {"width": 390, "height": 800}, "color_scheme": "dark"})):
            extra = browser.new_page(**kw)
            extra.goto(root)
            extra.screenshot(path=os.path.join(out, f"{name}.png"), full_page=True)
        browser.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
