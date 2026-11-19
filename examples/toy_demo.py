"""Synthetic numerical demonstration, not evidence of LLM capability improvement."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from weight_space_lab.core import inspect, prepare, write_candidate

root = Path("runs/toy").resolve()
root.mkdir(parents=True, exist_ok=True)
source = root / "model.safetensors"
if source.exists():
    raise SystemExit("runs/toy exists; choose a fresh workspace to repeat")
save_file({"mlp.down_proj.weight": np.eye(4, dtype=np.float32)}, str(source))
plan = {
    "checkpoint": source.name,
    "source_sha256": inspect(root, source.name)["sha256"],
    "hypothesis": "Synthetic example: attenuate coordinate zero in output column zero.",
    "limits": {"max_changed_fraction": .1, "max_relative_l2": .2},
    "operations": [{"tensor": "mlp.down_proj.weight", "kind": "project_out",
                    "axis": 1, "indices": [0], "direction": [1, 0, 0, 0], "strength": .25}],
}
(root / "plan.json").write_text(json.dumps(plan, indent=2))
print(json.dumps(prepare(root, plan)[1], indent=2))
print(json.dumps(write_candidate(root, plan, "candidate-001"), indent=2))
