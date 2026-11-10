import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.gsd_overnight_report import ARMS, check_prerequisites, rows


class QueueChecks(unittest.TestCase):
    def prepare(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        write(root / "pipeline_state.json", {"status": "complete"})
        write(root / "source_40000/source_training_report.json", {"completed": True, "last_observation": {"iteration": 40000}})
        write(root / "shared_warm/seed1/warm/complete.json", {"iteration": 20000, "checkpoint_sha256": "warm"})
        write(root / "shared_warm/seed1/protocol.json", {"source_checkpoint_sha256": "source", "manifests": {"target_train": "data"}})
        for arm in ARMS:
            path = root / "pilot" / arm / "seed1"
            write(path / "snd/complete.json", {"iteration": 1500})
            write(path / "protocol.json", {"smoke": False, "gsd_study": {"arm": arm, "coverage": .5, "lambda": 1},
                "gsd_scope": {"main_stop_updates": 1500}, "warm_checkpoint_sha256": "warm", "source_checkpoint_sha256": "source",
                "manifests": {"target_train": "data"}, "recipe": {"batch_size": 6, "snd": {"lr_max_iter": 32000}}})
            (path / "snd/training_history.jsonl").write_text('{"loss": 1}\n')
    def test_dependency_and_lineage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            self.assertEqual(len(check_prerequisites(root)["verified_pilots"]), 4)
            p = root / "pilot/generic/seed1/protocol.json"
            data = json.loads(p.read_text())
            data["warm_checkpoint_sha256"] = "another_warm"
            p.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                check_prerequisites(root)
    def test_waits_for_successful_pipeline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            (root / "pipeline_state.json").write_text('{"status":"failed"}')
            with self.assertRaises(RuntimeError):
                check_prerequisites(root)
    def test_partial_last_json_line(self):
        with TemporaryDirectory() as directory:
            p = Path(directory) / "history.jsonl"
            p.write_text('{"iteration":20}\n{"iteration":')
            self.assertEqual(rows(p), [{"iteration": 20}])


if __name__ == "__main__":
    unittest.main()
