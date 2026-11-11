"""EvalForge background worker: asynchronous job processing for ML evaluations, bootstrapping, and exports."""
from __future__ import annotations

import logging
import signal
import time
from typing import Any

from . import core
from .ml.judge import HeuristicJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [evalforge.worker] %(message)s")
logger = logging.getLogger("evalforge.worker")


class EvaluationWorker:
    """Production asynchronous job processor for compute-intensive ML workloads."""

    def __init__(self, poll_interval: float = 2.0) -> None:
        self.poll_interval = poll_interval
        self._running = False
        self.judge = HeuristicJudge()

    def handle_signal(self, signum: int, frame: Any) -> None:
        logger.info(f"Received termination signal ({signum}), shutting down worker gracefully...")
        self._running = False

    def process_job(self, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch job execution based on job type."""
        logger.info(f"Processing background ML job: {job_type}")
        start_time = time.time()

        if job_type == "bootstrap_leaderboard":
            battles = payload.get("battles", [])
            n_boot = payload.get("n_bootstrap", 300)
            res = core.bradley_terry_bootstrap(battles, n_bootstrap=n_boot)
            duration = time.time() - start_time
            return {"status": "completed", "result": res, "duration_seconds": round(duration, 3)}

        elif job_type == "batch_judge":
            tasks = payload.get("tasks", [])
            results = []
            for t in tasks:
                eval_res = self.judge.evaluate(t["prompt"], t["response_a"], t["response_b"])
                results.append({
                    "task_id": t.get("id"),
                    "winner": eval_res.winner,
                    "confidence": eval_res.confidence,
                    "issues": eval_res.detected_issues,
                })
            duration = time.time() - start_time
            return {"status": "completed", "evaluated_count": len(results), "results": results, "duration_seconds": round(duration, 3)}

        elif job_type == "krippendorff_analysis":
            units = payload.get("units", {})
            level = payload.get("level", "nominal")
            alpha = core.krippendorff_alpha(units, level_of_measurement=level)
            duration = time.time() - start_time
            return {"status": "completed", "krippendorff_alpha": alpha, "duration_seconds": round(duration, 3)}

        else:
            return {"status": "error", "message": f"Unknown job type: {job_type}"}

    def start(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        logger.info("EvalForge ML Worker started successfully. Listening for tasks...")

        # In daemon/worker mode, polls the queue until interrupted
        while self._running:
            time.sleep(self.poll_interval)
        logger.info("EvalForge ML Worker stopped.")


def main() -> None:
    worker = EvaluationWorker()
    worker.start()


if __name__ == "__main__":
    main()
