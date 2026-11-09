import json
import os
import re
import tempfile
import unittest
from unittest import mock

from evalforge import core
from evalforge.server import create_app

KEY = {"X-API-Key": "test-key"}


class CoreTests(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(core.cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(core.cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)
        self.assertEqual(core.cohens_kappa([1, 1], [1, 1]), 1.0)
        with self.assertRaises(ValueError):
            core.cohens_kappa([], [])

    def test_pairwise_and_overlap_threshold(self):
        by = {"a": {i: "a" for i in range(6)}, "b": {i: "a" for i in range(6)}, "c": {0: "a"}}
        out = core.pairwise_kappas(by)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["a"], out[0]["b"], out[0]["n"]), ("a", "b", 6))

    def test_gold_and_flagging(self):
        self.assertEqual(core.gold_accuracy({"q1": "A", "q2": "B", "q3": "A"}, {"q1": "A", "q2": "A"}), 0.5)
        self.assertEqual(core.gold_accuracy({}, {"q": "A"}), 0.0)
        self.assertEqual(core.flag_unreliable({"r1": 0.9, "r2": 0.5}), ["r2"])

    def test_consensus(self):
        self.assertEqual(core.consensus(["a", "a", "b"]), "a")
        self.assertIsNone(core.consensus(["a", "b"]))
        self.assertIsNone(core.consensus(["a"]))
        self.assertIsNone(core.consensus(["tie", "tie", "a"]))

    def test_swap_is_deterministic_and_balanced(self):
        self.assertEqual(core.swap_for(1, "x"), core.swap_for(1, "x"))
        share = sum(core.swap_for(i, "x") for i in range(400)) / 400
        self.assertTrue(0.4 < share < 0.6)

    def test_bradley_terry(self):
        p = core.bradley_terry([("A", "B")] * 8 + [("B", "A")] * 2 + [("B", "C")] * 6 + [("C", "B")] * 2)
        self.assertTrue(p["A"] > p["B"] > p["C"])
        self.assertAlmostEqual(sum(p.values()), 3.0, places=6)
        self.assertGreater(core.elo_scale(p["A"]), core.elo_scale(p["C"]))

    def test_dpo(self):
        rows = core.to_dpo([{"prompt": "p", "a": "x", "b": "y", "winner": "b"},
                            {"prompt": "p", "a": "x", "b": "y", "winner": "tie"}])
        self.assertEqual(rows, [{"prompt": "p", "chosen": "y", "rejected": "x"}])


def task(i, gold=None, prompt=None):
    return {"prompt": prompt or f"prompt {i}", "model_a": "m1", "response_a": f"good answer {i}",
            "model_b": "m2", "response_b": f"weak answer {i}", "gold": gold}


def csrf(html):
    return re.search(r'name="_csrf" value="([^"]+)"', html).group(1)


def login(client, rater):
    token = csrf(client.get("/annotate").get_data(as_text=True))
    return client.post("/session", data={"_csrf": token, "rater": rater})


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"API_KEY": "test-key"})
        self.env.start()
        self.app = create_app(os.path.join(tempfile.mkdtemp(), "t.db"), {"RATE_LIMIT": 10000})
        self.c = self.app.test_client()

    def tearDown(self):
        self.env.stop()

    def import_tasks(self, tasks):
        return self.c.post("/api/tasks", json=tasks, headers=KEY)

    def annotate_all(self, client, rater, pick="a", issues=()):
        """Annotate every task so that the canonical winner is `pick` (handles per-rater position swap)."""
        self.assertEqual(login(client, rater).status_code, 302)
        for _ in range(100):
            page = client.get("/annotate").get_data(as_text=True)
            if "All done" in page:
                return
            tid = int(re.search(r'name="task_id" value="(\d+)"', page).group(1))
            flip = core.swap_for(tid, rater)
            choice = ("second" if flip else "first") if pick == "a" else ("first" if flip else "second")
            r = client.post("/annotate", data={"_csrf": csrf(page), "task_id": tid, "choice": choice,
                                               "note": "n", "issues": list(issues)})
            self.assertEqual(r.status_code, 302)

    def test_import_auth_and_validation(self):
        self.assertEqual(self.import_tasks([task(1)]).status_code, 201)
        self.assertEqual(self.c.post("/api/tasks", json=[task(1)]).status_code, 401)
        with mock.patch.dict(os.environ, {"API_KEY": ""}):
            self.assertEqual(self.import_tasks([task(1)]).status_code, 503)
        for bad in ([], {}, [{"prompt": "x"}], [task(1, gold="z")], ["str"], [dict(task(1), prompt="")]):
            self.assertEqual(self.import_tasks(bad).status_code, 400, bad)

    def test_full_workflow_consensus_leaderboard_and_export(self):
        self.import_tasks([task(i, gold="a" if i < 4 else None) for i in range(8)])
        for rater in ("alice", "bob", "carol"):
            self.annotate_all(self.app.test_client(), rater, pick="a")
        stats = self.c.get("/api/stats").get_json()
        self.assertEqual((stats["tasks"], stats["annotations"]), (8, 24))
        self.assertEqual(stats["leaderboard"][0]["model"], "m1")
        self.assertAlmostEqual(stats["mean_kappa"], 1.0)
        out = self.c.get("/api/export/dpo", headers=KEY).get_data(as_text=True).strip().splitlines()
        rows = [json.loads(line) for line in out]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["chosen"].startswith("good") for r in rows))
        for path in ("/", "/raters", "/leaderboard"):
            self.assertEqual(self.c.get(path).status_code, 200)

    def test_issue_tags_are_attributed_to_the_dispreferred_model(self):
        self.import_tasks([task(i) for i in range(4)])
        for rater in ("alice", "bob"):
            self.annotate_all(self.app.test_client(), rater, pick="a", issues=("factual_error", "bias"))
        stats = self.c.get("/api/stats").get_json()
        self.assertEqual({i["label"]: i["count"] for i in stats["issues"]}, {"Factual error": 8, "Bias": 8})
        board = {m["model"]: m for m in stats["leaderboard"]}
        self.assertEqual((board["m2"]["issues"], board["m1"]["issues"]), (16, 0))
        self.assertIn(b"Factual error", self.c.get("/").data)

    def test_unreliable_rater_flagged_by_gold(self):
        self.import_tasks([task(i, gold="a") for i in range(6)])
        self.annotate_all(self.app.test_client(), "good", pick="a")
        self.annotate_all(self.app.test_client(), "bad", pick="b")
        raters = {r["name"]: r for r in self.c.get("/api/stats").get_json()["raters"]}
        self.assertFalse(raters["good"]["flagged"])
        self.assertTrue(raters["bad"]["flagged"])
        self.assertIn(b"review", self.c.get("/").data)

    def test_csrf_and_input_validation(self):
        self.import_tasks([task(1)])
        c = self.app.test_client()
        token = csrf(c.get("/annotate").get_data(as_text=True))
        self.assertEqual(c.post("/session", data={"_csrf": "forged", "rater": "alice"}).status_code, 403)
        self.assertEqual(self.app.test_client().post("/session", data={"rater": "alice"}).status_code, 403)
        self.assertEqual(c.post("/session", data={"_csrf": token, "rater": "../etc"}).status_code, 400)
        self.assertEqual(c.post("/annotate", data={"_csrf": token, "task_id": 1, "choice": "first"}).status_code, 400)  # no rater yet
        self.assertEqual(c.get("/annotate?rater=mallory").status_code, 200)
        self.assertNotIn(b"mallory", c.get("/annotate").data)  # GET never changes the session
        self.assertEqual(c.post("/session", data={"_csrf": token, "rater": "alice"}).status_code, 302)
        base = {"_csrf": token, "task_id": 1, "choice": "first"}
        self.assertEqual(c.post("/annotate", data={**base, "_csrf": "forged"}).status_code, 403)
        self.assertEqual(c.post("/annotate", data={**base, "choice": "evil"}).status_code, 400)
        self.assertEqual(c.post("/annotate", data={**base, "issues": "not_a_tag"}).status_code, 400)
        self.assertEqual(c.post("/annotate", data={**base, "task_id": "x"}).status_code, 400)
        self.assertEqual(c.post("/annotate", data={**base, "task_id": 99}).status_code, 404)
        self.assertEqual(c.post("/annotate", data={**base, "note": "x" * 1500}).status_code, 302)  # truncated

    def test_switch_rater(self):
        self.import_tasks([task(1)])
        c = self.app.test_client()
        login(c, "alice")
        page = c.get("/annotate").get_data(as_text=True)
        self.assertIn("alice", page)
        c.post("/session/end", data={"_csrf": csrf(page)})
        self.assertIn("Start annotating", c.get("/annotate").get_data(as_text=True))

    def test_resubmission_updates_and_refreshes_cached_analytics(self):
        self.import_tasks([task(1)])
        c = self.app.test_client()
        login(c, "z")
        token = csrf(c.get("/annotate").get_data(as_text=True))
        for choice in ("first", "second"):
            c.post("/annotate", data={"_csrf": token, "task_id": 1, "choice": choice})
            self.assertEqual(self.c.get("/api/stats").get_json()["annotations"], 1)
        flip = core.swap_for(1, "z")
        expected = "a" if (choice == "second") == flip else "b"  # last choice wins
        for rater in ("y",):
            self.annotate_all(self.app.test_client(), rater, pick=expected)
        self.assertEqual(self.c.get("/api/stats").get_json()["annotations"], 2)

    def test_analytics_cached_until_data_changes(self):
        self.import_tasks([task(i) for i in range(3)])
        with mock.patch("evalforge.server.core.pairwise_kappas", wraps=core.pairwise_kappas) as spy:
            for path in ("/", "/raters", "/leaderboard", "/metrics", "/api/stats"):
                self.c.get(path)
            self.assertEqual(spy.call_count, 1)
            self.annotate_all(self.app.test_client(), "q", pick="a")
            self.c.get("/")
            self.assertEqual(spy.call_count, 2)

    def test_export_requires_key_and_pages_empty_state(self):
        self.assertEqual(self.c.get("/api/export/dpo").status_code, 401)
        for path in ("/", "/raters", "/leaderboard", "/annotate"):
            self.assertEqual(self.c.get(path).status_code, 200)

    def test_xss_escaped_and_sql_inert(self):
        self.import_tasks([task(1, prompt="<script>alert(1)</script> '); DROP TABLE tasks;--")])
        c = self.app.test_client()
        login(c, "eve")
        page = c.get("/annotate")
        self.assertNotIn(b"<script>alert(1)", page.data)
        self.assertIn(b"&lt;script&gt;", page.data)
        self.assertEqual(self.c.get("/api/stats").get_json()["tasks"], 1)
        self.assertEqual(page.headers["X-Frame-Options"], "DENY")

    def test_session_cookie_flags_metrics(self):
        cookie = self.app.test_client().get("/annotate").headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        with mock.patch.dict(os.environ, {"COOKIE_SECURE": "1"}):
            secure = create_app(os.path.join(tempfile.mkdtemp(), "s.db")).test_client().get("/annotate")
        self.assertIn("Secure", secure.headers["Set-Cookie"])
        self.assertIn("evalforge_tasks_total", self.c.get("/metrics").data.decode())


if __name__ == "__main__":
    unittest.main()
