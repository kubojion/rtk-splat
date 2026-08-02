import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

import rtk_splat
import rtk_splat.core


PACKAGE = Path(rtk_splat.__file__).resolve().parent
CORE = Path(rtk_splat.core.__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
FORBIDDEN_IMPORT_ROOTS = {
    "adapters",
    "backends",
    "diagnostics",
    "rclpy",
    "rosbags",
    "rospy",
    "workflows",
}
FORBIDDEN_PACKAGE_PREFIXES = tuple(
    f"rtk_splat.{name}" for name in (
        "adapters",
        "backends",
        "diagnostics",
        "frontends",
        "workflows",
    )
)
LEGACY_TOP_LEVEL_PACKAGES = (
    "adapters",
    "backends",
    "diagnostics",
    "frontends",
    "workflows",
)
MOVED_MODULES = {
    "bagio.py",
    "calibration_io.py",
    "calibration_sidecar.py",
    "cameras.py",
    "cli.py",
    "colmap_global.py",
    "colmap_stereo.py",
    "configio.py",
    "ingest_agrigs.py",
    "metric_calibration.py",
    "pose_sources.py",
    "train.py",
}


class CoreIsolationTests(unittest.TestCase):
    def test_pose_sources_do_not_import_the_concrete_ros2_adapter(self):
        pose_sources = (PACKAGE / "adapters" / "pose_sources.py").read_text()
        rtk_io = (PACKAGE / "adapters" / "ros2_rtk_io.py").read_text()
        self.assertNotIn("ros2_zed_ublox", pose_sources)
        self.assertNotIn("pose_sources", rtk_io)

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
                    if (
                        name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS
                        or any(
                            name == prefix or name.startswith(prefix + ".")
                            for prefix in FORBIDDEN_PACKAGE_PREFIXES
                        )
                    ):
                        violations.append(f"{path.name}:{node.lineno}: {name}")
        self.assertEqual(violations, [])

    def test_only_rtk_splat_is_an_installed_top_level_namespace(self):
        project = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        package_section = project.split(
            "[tool.setuptools.packages.find]", maxsplit=1
        )[1].split("\n[tool.", maxsplit=1)[0]
        self.assertIn('package-dir = {"" = "src"}', project)
        self.assertIn('where = ["src"]', package_section)
        self.assertIn('include = ["rtk_splat*"]', package_section)
        for name in LEGACY_TOP_LEVEL_PACKAGES:
            self.assertNotIn(f'"{name}*"', package_section)
        present = [
            name for name in LEGACY_TOP_LEVEL_PACKAGES
            if (REPOSITORY / name).exists()
        ]
        self.assertEqual(present, [])

    def test_package_uses_src_layout(self):
        self.assertTrue((SOURCE / "rtk_splat" / "__init__.py").is_file())
        self.assertFalse((REPOSITORY / "rtk_splat").exists())

    def test_runtime_modules_do_not_publish_checkout_root_constants(self):
        from rtk_splat.core import verify_golden
        from rtk_splat.workflows import cli

        self.assertFalse(hasattr(cli, "REPO_ROOT"))
        self.assertFalse(hasattr(verify_golden, "REPOSITORY_ROOT"))

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
                importlib.import_module(f"rtk_splat.core.{name}")

    def test_diagnostics_have_no_ros_runtime_imports(self):
        violations = []
        for path in sorted((PACKAGE / "diagnostics").glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    if name.split(".", 1)[0] in {"rclpy", "rosbags", "rospy"}:
                        violations.append(f"{path.name}:{node.lineno}: {name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
