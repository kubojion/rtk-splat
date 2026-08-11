import tempfile
import unittest
from pathlib import Path

from test_segment_contract import _write_valid

from rtk_splat.core.segment import SegmentContractError, SegmentReader
from rtk_splat.workflows.segment_transfer import (
    TRANSFER_MANIFEST,
    materialize_portable_segment,
    verify_portable_segment,
)
from rtk_splat.workflows import cli


class PortableSegmentTests(unittest.TestCase):
    def test_cli_requires_and_scopes_the_portable_destination(self):
        parsed = cli.build_parser().parse_args(
            [
                "segment-materialize",
                "--config",
                "/unused.yaml",
                "--portable-segment",
                "/portable/segment",
                "--link-mode",
                "hardlink",
            ]
        )
        overrides = cli._stage_overrides(parsed)
        self.assertEqual(overrides["portable_segment"], "/portable/segment")
        self.assertEqual(overrides["link_mode"], "hardlink")
        with self.assertRaisesRegex(ValueError, "requires --portable-segment"):
            cli.main(
                ["segment-materialize", "--config", "/does/not/exist.yaml"]
            )
        with self.assertRaisesRegex(ValueError, "valid only"):
            cli.main(
                [
                    "validate",
                    "--config",
                    "/does/not/exist.yaml",
                    "--portable-segment",
                    "/portable/segment",
                ]
            )

    def test_materializes_symlinked_assets_and_verifies_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_valid(root / "source")
            external = root / "external-left.jpg"
            original = source.root / "images/left_0.jpg"
            external.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(external)
            self.assertEqual(SegmentReader(source.root).validate().root, source.root)

            portable = materialize_portable_segment(
                source.root, root / "portable", link_mode="auto"
            )
            evidence = verify_portable_segment(portable)

            self.assertTrue(evidence["verified"])
            self.assertEqual(evidence["n_frames"], 3)
            self.assertTrue((portable / TRANSFER_MANIFEST).is_file())
            self.assertFalse(
                any(path.is_symlink() for path in portable.rglob("*"))
            )
            self.assertEqual(
                (portable / "images/left_0.jpg").read_bytes(),
                external.read_bytes(),
            )

    def test_tampered_or_extra_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_valid(root / "source")
            portable = materialize_portable_segment(
                source.root, root / "portable", link_mode="copy"
            )
            image = portable / "images/left_0.jpg"
            image.write_bytes(image.read_bytes() + b"tamper")
            with self.assertRaisesRegex(
                SegmentContractError, "size mismatch|SHA-256 mismatch"
            ):
                verify_portable_segment(portable)

            other = materialize_portable_segment(
                source.root, root / "portable-extra", link_mode="copy"
            )
            (other / "unexpected.txt").write_text("not sealed")
            with self.assertRaisesRegex(SegmentContractError, "file set differs"):
                verify_portable_segment(other)

    def test_refuses_existing_destination_and_invalid_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_valid(root / "source")
            destination = root / "portable"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                materialize_portable_segment(source.root, destination)
            with self.assertRaisesRegex(ValueError, "link_mode"):
                materialize_portable_segment(
                    source.root, root / "unused", link_mode="symlink"
                )


if __name__ == "__main__":
    unittest.main()
