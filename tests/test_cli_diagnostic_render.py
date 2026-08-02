import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from rtk_splat.workflows import cli


class DiagnosticRenderCliTests(unittest.TestCase):
    @staticmethod
    def _args(**updates):
        values = {
            "backend": "global",
            "backend_name": "backend",
            "feature_profile": None,
            "frontend_name": None,
            "keyframe_preset": None,
            "pose_name": "pose-diagnostic-render",
            "refinement_name": None,
            "allow_failed_georeferencing_for_render": True,
        }
        values.update(updates)
        return NS(**values)

    def test_backend_export_receives_explicit_diagnostic_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = NS(paths=NS(workdir=root), mapper=NS())
            expected = root / "pose_artifacts/pose-diagnostic-render"
            with mock.patch(
                "rtk_splat.backends.mapper.export_pose_artifact",
                return_value=expected,
            ) as export:
                cli.cmd_backend_export(cfg, self._args())

            self.assertTrue(
                export.call_args.kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertEqual(export.call_args.args[1], "pose-diagnostic-render")

    def test_cloud_and_train_receive_explicit_diagnostic_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reader = NS(root=root / "segment")
            cfg = NS(
                paths=NS(workdir=root),
                pose=NS(artifact="production"),
                cloud=NS(artifact_root=root / "cloud_artifacts"),
                train=NS(run_name="production"),
            )
            args = self._args(run_name="run-diagnostic-render")
            cloud_file = root / "cloud_artifacts/pose-diagnostic-render/init_cloud.npz"

            with (
                mock.patch.object(cli, "_reader", return_value=reader),
                mock.patch(
                    "rtk_splat.workflows.cloud.construct_initial_cloud",
                    return_value=(cloud_file, 10),
                ) as cloud,
            ):
                cli.cmd_cloud(cfg, args)
            self.assertTrue(
                cloud.call_args.kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )

            with (
                mock.patch.object(cli, "_reader", return_value=reader),
                mock.patch(
                    "rtk_splat.workflows.runtime_config.resolve_training_controls"
                ),
                mock.patch(
                    "rtk_splat.backends.gsplat.train_tile", return_value={}
                ) as train,
            ):
                cli.cmd_train(cfg, args)
            self.assertTrue(
                train.call_args.kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertEqual(cfg.pose.artifact, "pose-diagnostic-render")
            self.assertEqual(cfg.train.run_name, "run-diagnostic-render")


if __name__ == "__main__":
    unittest.main()
