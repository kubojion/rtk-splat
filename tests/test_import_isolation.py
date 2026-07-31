import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

import rtk_splat


CORE = Path(rtk_splat.__file__).resolve().parent
FORBIDDEN_IMPORT_ROOTS = {
    "adapters",
    "backends",
    "diagnostics",
    "rclpy",
    "rosbags",
    "rospy",
    "workflows",
}
MOVED_MODULES = {
    "bagio.py",
    "calibration_io.py",
    "calibration_sidecar.py",
    "cameras.py",
    "cli.py",
    "colmap_global.py",
    "colmap_stereo.py",
    "ingest_agrigs.py",
    "metric_calibration.py",
    "pose_sources.py",
    "train.py",
}


class CoreIsolationTests(unittest.TestCase):
    def test_core_has_no_ros_dataset_or_backend_imports(self):
        violations = []
        for path in sorted(CORE.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{path.name}:{node.lineno}: {name}")
        self.assertEqual(violations, [])

    def test_old_flat_modules_are_deleted(self):
        present = sorted(path.name for path in CORE.glob("*.py") if path.name in MOVED_MODULES)
        self.assertEqual(present, [])

    def test_core_stays_within_size_budget(self):
        line_count = sum(
            len(path.read_text().splitlines()) for path in CORE.glob("*.py")
        )
        self.assertLessEqual(line_count, 2_000)

    def test_every_core_module_imports_without_adapter_runtime(self):
        names = [
            module.name
            for module in pkgutil.iter_modules([str(CORE)])
            if not module.name.startswith("_")
        ]
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(f"rtk_splat.{name}")


if __name__ == "__main__":
    unittest.main()
