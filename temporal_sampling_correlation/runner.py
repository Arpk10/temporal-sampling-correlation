import json
from pathlib import Path

import pandas as pd

from .sampler import generate_condition


def main():
    task_path = Path("/tmp/tsc_task.json")
    if not task_path.exists():
        raise RuntimeError("Missing /tmp/tsc_task.json")

    task = json.loads(task_path.read_text())

    rows = generate_condition(
        prompt=task["prompt"],
        rho=float(task["rho"]),
        seeds=[int(seed) for seed in task["seeds"]],
        n_new=int(task["n_new"]),
        device=task.get("device", "cuda"),
        model_name=task["model_name"],
        temperature=float(task["temperature"]),
        top_p=float(task["top_p"]),
        prompt_id=int(task["prompt_id"]),
    )

    out = Path("/tmp/tsc_result.json")
    out.write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv("/tmp/tsc_result.csv", index=False)

    print(json.dumps({
        "rows": len(rows),
        "prompt_id": task["prompt_id"],
        "rho": task["rho"],
        "seeds": task["seeds"],
    }))


if __name__ == "__main__":
    main()
