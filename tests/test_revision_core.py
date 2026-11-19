"""Fast revision-contract tests. No solver, training, or manuscript data changes."""
from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import revision_cli as cli
import revision_inputs as inputs


class RevisionCoreTest(unittest.TestCase):
    def test_real_config_plan_and_seed_schedule(self):
        for flag, counts in (([], [3, 3, 3]), (["--full-matrix"], [3, 6, 6])):
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = cli.main(["--stage", "plan", "--config",str(PROJECT/"config_v3.yaml"), *flag])
            self.assertEqual(code, 0)
            plan = json.loads(stream.getvalue())
            schedule = plan["training_schedule"]
            self.assertEqual(list(schedule), ["20260923", "20270923", "20280923"])
            self.assertEqual(list(schedule.values())[0], list(cli.ABLATIONS))
            self.assertEqual([len(x) for x in schedule.values()], counts)
            self.assertEqual(len(set(plan["paired_optimizer_seeds"])), 3)
            self.assertEqual(plan["frozen_model_build_seed"], 20260923)
            self.assertEqual(set(cli.strategies(0, bool(flag), True)), set(cli.ORIGINAL + cli.ABLATIONS))

    def test_summary_requires_pairs_and_reports_sample_sd_only_for_repeats(self):
        # Synthetic unit test; never copied into any scientific results directory.
        seeds = [20260923, 20270923, 20280923]
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            r = Path(temp)
            for i, seed in enumerate(seeds):
                d = r / f"repeat_{i:03d}" / "revision_evaluation"
                d.mkdir(parents=True)
                for s in cli.strategies(i, False, True):
                    rmse = (2 + i) if s == "random" else 1
                    m = [{**dict.fromkeys(cli.METRICS, 0.25), "rmse_R":rmse, "split":role,
                          "n_geometries":2, "n_states":6} for role in ("iid", "ood")]
                    (d/f"{s}.json").write_text(json.dumps({"seed":seed,"strategy":s,"metrics":m}))
            answer = cli.summarize(r, seeds, False)
            with (Path(answer["directory"])/"paired_summary.csv").open(encoding="utf-8-sig") as f:
                rows = {(v["comparison"],v["split"]):v for v in csv.DictReader(f)}
            random = rows["random", "ood"]
            self.assertEqual(int(random["paired_seeds"]), 3)
            self.assertAlmostEqual(float(random["mean_full_minus_comparison"]), -2.)
            self.assertAlmostEqual(float(random["sample_sd"]), 1.)
            for a in cli.ABLATIONS:
                self.assertEqual(int(rows[a,"ood"]["paired_seeds"]), 1)
                self.assertEqual(rows[a,"ood"]["sample_sd"], "")
            (r/"repeat_002"/"revision_evaluation"/"random.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                cli.summarize(r, seeds, False)

    def test_same_size_input_mutation_is_detected_after_copy(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            data = root/"inputs"; data.mkdir()
            label = data/"fixed_test_fixture"; label.write_bytes(b"v1")
            snap = {"files":{"fixed_test_fixture":{"bytes":2,"sha256":hashlib.sha256(b"v1").hexdigest()}}}
            snap["signature"] = inputs._fingerprint(snap)
            (data/inputs.SNAPSHOT).write_text(json.dumps(snap),encoding="utf-8")
            self.assertEqual(inputs.verify_snapshot(data), snap)
            target = root/"repeat_001"/"fixed_test_fixture"
            digest = snap["files"]["fixed_test_fixture"]["sha256"]
            inputs._install_file(label, target, digest)
            target.write_bytes(b"v2")  # ordinary copies, no shared data inode
            self.assertEqual(label.read_bytes(), b"v1")
            with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                inputs._install_file(label, target, digest)
            self.assertEqual(target.read_bytes(), b"v2")
            label.write_bytes(b"x1")
            with self.assertRaisesRegex(RuntimeError,"changed"):
                inputs.verify_snapshot(data)


if __name__ == "__main__":
    unittest.main()
