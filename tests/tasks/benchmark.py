"""τ-bench style K^trial benchmark runner with pass@k metric."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from test_01_create_fasta_for_favorite_protein import run_task as fasta_run_task
from testutils import K_TRIAL, TESTDB1_DEV_DIR, TESTDB1_NAME, TESTDB1_STORAGE, pass_at_k


def _reset_lamindb(run_dir: Path) -> None:
    """Wipe trial dir and reinit a fresh lamindb instance for the next trial."""
    import lamindb as ln
    from laminagent._setup import setup as laminagent_setup

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    storage_root = Path(TESTDB1_STORAGE)
    if storage_root.exists():
        shutil.rmtree(storage_root)
        ln.setup.delete(TESTDB1_NAME, force=True)

    ln.setup.init(name=TESTDB1_NAME, storage=storage_root, modules="bionty")
    ln.setup.settings.dev_dir = run_dir
    laminagent_setup(verbose=False)


TASKS: dict[str, object] = {
    "01_create_fasta_for_favorite_protein": fasta_run_task,
}
K_VALUES = [1, 2]


def run_benchmark(k_trial: int = K_TRIAL, k_values: list[int] = K_VALUES) -> None:
    results: dict[str, int] = {}

    for task_name, run_task in TASKS.items():
        successes = 0
        print(f"\n[{task_name}] running {k_trial} trials...")
        for trial_idx in range(k_trial):
            run_dir = Path(TESTDB1_DEV_DIR) / task_name / f"trial_{trial_idx:02d}"

            # clean slate: wipe trial dir, delete and reinit lamindb instance
            _reset_lamindb(run_dir)

            try:
                run_task(run_dir)  # type: ignore[operator]
                passed = True
            except Exception as exc:
                print(f"  error: {exc}")
                passed = False
            print(f"  trial {trial_idx:02d}: {'PASS' if passed else 'FAIL'}")
            successes += int(passed)

        results[task_name] = successes

    console = Console()
    table = Table(title=f"pass@k  (K^trial = {k_trial})")
    table.add_column("Task", style="cyan")
    table.add_column("successes")
    for k in k_values:
        if k <= k_trial:
            table.add_column(f"pass@{k}")

    for task_name, successes in results.items():
        row = [task_name, f"{successes}/{k_trial}"]
        for k in k_values:
            if k <= k_trial:
                row.append(f"{pass_at_k(k_trial, successes, k):.3f}")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="τ-bench K^trial benchmark")
    parser.add_argument("--k_trial", type=int, default=K_TRIAL)
    parser.add_argument("--k_values", type=int, nargs="+", default=K_VALUES)
    args = parser.parse_args()
    run_benchmark(k_trial=args.k_trial, k_values=args.k_values)
