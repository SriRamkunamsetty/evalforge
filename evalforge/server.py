"""Flask app: human preference annotation, rater QA, issue tagging, leaderboard, and enterprise ML/DPO export."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing

from flask import Flask, Response, abort, g, jsonify, redirect, render_template, request, session

from . import core
from .ml.export import DatasetFormat, export_preference_dataset
from .ml.judge import HeuristicJudge
from .security import check_csrf, init_security, require_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, 
  prompt TEXT NOT NULL,
  model_a TEXT NOT NULL, 
  response_a TEXT NOT NULL, 
  model_b TEXT NOT NULL, 
  response_b TEXT NOT NULL,
  gold TEXT CHECK (gold IN ('a','b'))
);
CREATE TABLE IF NOT EXISTS annotations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, 
  rater TEXT NOT NULL,
  winner TEXT NOT NULL CHECK (winner IN ('a','b','tie')), 
  issues TEXT NOT NULL DEFAULT '', 
  note TEXT,
  ts REAL NOT NULL, 
  rev INTEGER NOT NULL DEFAULT 0, 
  UNIQUE(task_id, rater)
);
CREATE TABLE IF NOT EXISTS judge_evaluations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  judge_name TEXT NOT NULL,
  winner TEXT NOT NULL CHECK (winner IN ('a','b','tie')),
  confidence REAL NOT NULL,
  rationale TEXT,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ann_rater ON annotations(rater);
CREATE INDEX IF NOT EXISTS idx_judge_task ON judge_evaluations(task_id);
"""
ISSUES = {
    "factual_error": "Factual error",
    "awkward_phrasing": "Awkward phrasing",
    "bias": "Bias",
    "unsafe_content": "Unsafe or inappropriate",
    "off_topic": "Off topic or unhelpful",
    "other": "Other",
}
RATER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
MIN_GOLD = 3


def create_app(db_path: str | None = None, test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path or os.path.join(os.environ.get("DATA_DIR", "./data"), "evalforge.db")
    app.config["MAX_CONTENT_LENGTH"] = 2_000_000
    if test_config:
        app.config.update(test_config)
    init_security(app)
    os.makedirs(os.path.dirname(os.path.abspath(app.config["DB_PATH"])), exist_ok=True)
    with closing(sqlite3.connect(app.config["DB_PATH"])) as conn:
        conn.executescript(SCHEMA)
    cache: dict = {}
    heuristic_judge = HeuristicJudge()

    def db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"])
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys=ON")
        return g.db

    @app.teardown_appcontext
    def _close(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    # ---- analytics (cached until tasks or annotations change) ---------------------------------
    def fingerprint() -> tuple:
        ann = db().execute("SELECT COUNT(*), COALESCE(SUM(rev),0), COALESCE(MAX(id),0) FROM annotations").fetchone()
        return (*ann, db().execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    def analytics() -> dict:
        fp = fingerprint()
        if cache.get("fp") == fp:
            return cache["value"]
        tasks = {r["id"]: r for r in db().execute("SELECT * FROM tasks")}
        ann = db().execute("SELECT task_id, rater, winner, issues FROM annotations").fetchall()
        by_rater: dict[str, dict[int, str]] = {}
        votes: dict[int, list[str]] = {}
        issue_total: Counter = Counter()
        issue_by_model: dict[str, Counter] = {}
        for r in ann:
            by_rater.setdefault(r["rater"], {})[r["task_id"]] = r["winner"]
            votes.setdefault(r["task_id"], []).append(r["winner"])
            for tag in filter(None, r["issues"].split(",")):
                issue_total[tag] += 1
                if r["winner"] in ("a", "b"):  # tags describe the response the rater did not prefer
                    loser = tasks[r["task_id"]]["model_" + ("b" if r["winner"] == "a" else "a")]
                    issue_by_model.setdefault(loser, Counter())[tag] += 1
        gold = {t: row["gold"] for t, row in tasks.items() if row["gold"]}
        raters = []
        for name, answers in sorted(by_rater.items()):
            answered_gold = [t for t in gold if t in answers]
            acc = core.gold_accuracy(answers, gold)
            raters.append({
                "name": name,
                "n": len(answers),
                "gold_n": len(answered_gold),
                "gold_acc": acc,
                "flagged": len(answered_gold) >= MIN_GOLD and acc < 0.7,
            })
        battles = []
        consensus_count = 0
        for t, v in votes.items():
            winner = core.consensus(v)
            if winner:
                consensus_count += 1
                row = tasks[t]
                battles.append((row["model_" + winner], row["model_" + ("b" if winner == "a" else "a")]))
        strengths = core.bradley_terry(battles) if battles else {}
        board = []
        for model, strength in strengths.items():
            counts = issue_by_model.get(model, Counter())
            top = counts.most_common(1)
            board.append({
                "model": model,
                "strength": strength,
                "elo": core.elo_scale(strength),
                "wins": sum(1 for won, _ in battles if won == model),
                "battles": sum(1 for b in battles if model in b),
                "issues": sum(counts.values()),
                "top_issue": ISSUES[top[0][0]] if top else "–",
            })
        board.sort(key=lambda x: -x["strength"])
        kappas = core.pairwise_kappas(by_rater)

        # Advanced inter-annotator agreement: Krippendorff's alpha
        units = {}
        for r in ann:
            units.setdefault(r["task_id"], {})[r["rater"]] = r["winner"]
        krip_alpha = core.krippendorff_alpha(units) if units else 1.0

        value = {
            "tasks": tasks,
            "votes": votes,
            "raters": raters,
            "board": board,
            "strengths": strengths,
            "kappas": kappas,
            "mean_kappa": sum(k["kappa"] for k in kappas) / len(kappas) if kappas else None,
            "krippendorff_alpha": round(krip_alpha, 4),
            "consensus_count": consensus_count,
            "battles": battles,
            "issues": [{"label": ISSUES[t], "count": n} for t, n in issue_total.most_common()],
            "n_ann": len(ann),
        }
        cache.clear()
        cache.update(fp=fp, value=value)
        return value

    # ---- system endpoints ---------------------------------------------------------------------
    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.get("/metrics")
    def metrics():
        a = analytics()
        krip = a.get("krippendorff_alpha", 1.0)
        body = (
            f"evalforge_tasks_total {len(a['tasks'])}\n"
            f"evalforge_annotations_total {a['n_ann']}\n"
            f"evalforge_raters_total {len(a['raters'])}\n"
            f"evalforge_flagged_raters {sum(r['flagged'] for r in a['raters'])}\n"
            f"evalforge_consensus_tasks_total {a['consensus_count']}\n"
            f"evalforge_krippendorff_alpha {krip:.4f}\n"
        )
        return Response(body, mimetype="text/plain")

    # ---- UI -----------------------------------------------------------------------------------
    @app.get("/")
    def index():
        return render_template("index.html", a=analytics())

    @app.get("/raters")
    def raters_page():
        return render_template("raters.html", a=analytics())

    @app.get("/leaderboard")
    def leaderboard_page():
        return render_template("leaderboard.html", a=analytics())

    @app.get("/annotate")
    def annotate():
        rater = session.get("rater")
        if not rater:
            return render_template("annotate.html", rater=None, task=None)
        
        # Check if active learning queue prioritization is requested
        use_active = request.args.get("mode") == "active"
        row = None

        if use_active:
            a = analytics()
            pending = db().execute(
                "SELECT * FROM tasks WHERE id NOT IN (SELECT task_id FROM annotations WHERE rater=?)",
                (rater,)
            ).fetchall()
            if pending:
                task_dicts = [dict(r) for r in pending]
                prioritized = core.active_queue_prioritization(
                    task_dicts, a["strengths"], a["votes"]
                )
                if prioritized:
                    row = prioritized[0]
        else:
            row = db().execute(
                "SELECT * FROM tasks WHERE id NOT IN (SELECT task_id FROM annotations WHERE rater=?) ORDER BY id LIMIT 1",
                (rater,),
            ).fetchone()

        done = db().execute("SELECT COUNT(*) FROM annotations WHERE rater=?", (rater,)).fetchone()[0]
        total = db().execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        task = None
        if row:
            first, second = ("b", "a") if core.swap_for(row["id"], rater) else ("a", "b")
            task = {
                "id": row["id"],
                "prompt": row["prompt"],
                "first": row["response_" + first],
                "second": row["response_" + second],
            }
        return render_template("annotate.html", rater=rater, task=task, done=done, total=total, issues=ISSUES)

    @app.post("/session")
    def start_session():
        """Set the rater name (POST + CSRF: state changes never happen on GET)."""
        check_csrf()
        rater = request.form.get("rater", "")
        if not RATER_RE.match(rater):
            abort(400)
        session["rater"] = rater
        return redirect("/annotate")

    @app.post("/session/end")
    def end_session():
        check_csrf()
        session.pop("rater", None)
        return redirect("/annotate")

    @app.post("/annotate")
    def submit():
        check_csrf()
        rater = session.get("rater")
        if not rater:
            abort(400)
        choice, note = request.form.get("choice", ""), request.form.get("note", "")[:1000]
        tags = set(request.form.getlist("issues"))
        try:
            task_id = int(request.form.get("task_id", ""))
        except ValueError:
            abort(400)
        if choice not in ("first", "second", "tie") or not tags <= ISSUES.keys():
            abort(400)
        if db().execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone() is None:
            abort(404)
        flip = core.swap_for(task_id, rater)
        winner = "tie" if choice == "tie" else (("b" if choice == "first" else "a") if flip else
                                                 ("a" if choice == "first" else "b"))
        db().execute(
            "INSERT INTO annotations(task_id, rater, winner, issues, note, ts) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(task_id, rater) DO UPDATE SET winner=excluded.winner, issues=excluded.issues, "
            "note=excluded.note, ts=excluded.ts, rev=annotations.rev + 1",
            (task_id, rater, winner, ",".join(sorted(tags)), note, time.time()),
        )
        db().commit()
        return redirect("/annotate")

    # ---- API v1 -------------------------------------------------------------------------------
    @app.post("/api/tasks")
    @require_key
    def import_tasks():
        body = request.get_json(silent=True)
        if not isinstance(body, list) or not 0 < len(body) <= 1000:
            abort(400)
        rows = []
        for t in body:
            if not isinstance(t, dict):
                abort(400)
            vals = [t.get(k) for k in ("prompt", "model_a", "response_a", "model_b", "response_b")]
            if not all(isinstance(v, str) and 0 < len(v) <= 8000 for v in vals) or t.get("gold") not in (None, "a", "b"):
                abort(400)
            rows.append((*vals, t.get("gold")))
        db().executemany(
            "INSERT INTO tasks(prompt, model_a, response_a, model_b, response_b, gold) VALUES (?,?,?,?,?,?)", rows
        )
        db().commit()
        return jsonify(imported=len(rows)), 201

    @app.get("/api/stats")
    def api_stats():
        a = analytics()
        return jsonify(
            tasks=len(a["tasks"]),
            annotations=a["n_ann"],
            raters=a["raters"],
            mean_kappa=a["mean_kappa"],
            krippendorff_alpha=a["krippendorff_alpha"],
            leaderboard=a["board"],
            issues=a["issues"],
        )

    @app.get("/api/v1/leaderboard/bootstrap")
    def api_leaderboard_bootstrap():
        """Compute Bradley-Terry rankings with 95% bootstrap confidence intervals."""
        a = analytics()
        battles = a.get("battles", [])
        n_boot = min(int(request.args.get("n", 100)), 500)
        summary = core.bradley_terry_bootstrap(battles, n_bootstrap=n_boot)
        return jsonify(bootstrap_leaderboard=summary, battles_count=len(battles))

    @app.get("/api/v1/queue/prioritized")
    def api_prioritized_queue():
        """Retrieve pending tasks ordered by active learning entropy / uncertainty score."""
        a = analytics()
        task_list = [dict(r) for r in a["tasks"].values()]
        prioritized = core.active_queue_prioritization(
            task_list, a["strengths"], a["votes"]
        )
        return jsonify(prioritized_tasks=prioritized[:100], count=len(prioritized))

    @app.post("/api/v1/judge/evaluate")
    @require_key
    def api_judge_evaluate():
        """Automated LLM/Heuristic judge evaluating a prompt and candidate responses."""
        payload = request.get_json(silent=True) or {}
        prompt = payload.get("prompt")
        resp_a = payload.get("response_a")
        resp_b = payload.get("response_b")
        if not prompt or not resp_a or not resp_b:
            abort(400)
        res = heuristic_judge.evaluate(prompt, resp_a, resp_b)
        return jsonify({
            "winner": res.winner,
            "confidence": res.confidence,
            "rationale": res.rationale,
            "scores": [
                {"criterion": s.criterion, "score_a": s.score_a, "score_b": s.score_b}
                for s in res.scores
            ],
            "detected_issues": res.detected_issues,
            "position_consistent": res.position_consistent,
        })

    @app.get("/api/export/dpo")
    @require_key
    def export_dpo():
        a = analytics()
        records = []
        for t, v in a["votes"].items():
            winner = core.consensus(v)
            if winner:
                row = a["tasks"][t]
                records.append({
                    "prompt": row["prompt"],
                    "a": row["response_a"],
                    "b": row["response_b"],
                    "winner": winner,
                })
        lines = "".join(json.dumps(r) + "\n" for r in core.to_dpo(records))
        return Response(
            lines,
            mimetype="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=dpo.jsonl"},
        )

    @app.get("/api/v1/export/preferences")
    @require_key
    def export_preferences():
        """Enterprise multi-format preference exporter (DPO, KTO, ChatML, HuggingFace)."""
        fmt_str = request.args.get("format", "dpo").lower()
        try:
            fmt = DatasetFormat(fmt_str)
        except ValueError:
            fmt = DatasetFormat.DPO

        a = analytics()
        battle_records = []
        for t, v in a["votes"].items():
            winner = core.consensus(v)
            if winner:
                row = a["tasks"][t]
                battle_records.append({
                    "prompt": row["prompt"],
                    "model_a": row["model_a"],
                    "response_a": row["response_a"],
                    "model_b": row["model_b"],
                    "response_b": row["response_b"],
                    "winner": winner,
                    "task_id": row["id"],
                })

        dataset = export_preference_dataset(battle_records, output_format=fmt)
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dataset)
        filename = f"evalforge_dataset_{fmt.value}.jsonl"
        return Response(
            lines,
            mimetype="application/x-ndjson",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/openapi.json")
    def openapi_spec():
        """OpenAPI 3.0.3 specification."""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "EvalForge Enterprise API",
                "version": "1.0.0",
                "description": "Production-grade Human-in-the-Loop & LLM-as-a-Judge Evaluation Platform",
            },
            "paths": {
                "/healthz": {"get": {"summary": "Healthcheck probe", "responses": {"200": {"description": "OK"}}}},
                "/metrics": {"get": {"summary": "Prometheus metrics", "responses": {"200": {"description": "Text"}}}},
                "/api/tasks": {"post": {"summary": "Batch task import", "security": [{"ApiKeyAuth": []}]}},
                "/api/stats": {"get": {"summary": "Aggregated evaluation statistics"}},
                "/api/v1/leaderboard/bootstrap": {"get": {"summary": "Bradley-Terry Elo with 95% Confidence Intervals"}},
                "/api/v1/queue/prioritized": {"get": {"summary": "Active learning uncertainty-prioritized task queue"}},
                "/api/v1/judge/evaluate": {"post": {"summary": "Run LLM/Heuristic judge on responses"}},
                "/api/v1/export/preferences": {"get": {"summary": "Export preference dataset (DPO, KTO, ChatML, HF)"}},
            },
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
                }
            },
        }
        return jsonify(spec)

    return app
