from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from gaze.s3 import (
    S3Config,
    create_s3_pull_manifest,
    load_s3_config,
    mount_path_for_uri,
    pull_processed_from_manifest,
    serial_download_backup,
    serial_process_backup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class S3WorkflowTests(unittest.TestCase):
    def test_s3_config_and_static_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s3.json"
            path.write_text(json.dumps({"bucket_uri": "s3://example-bucket/gaze", "mount_root": "/mnt/s3"}))
            cfg = load_s3_config(path)
            self.assertEqual(load_s3_config().bucket_uri, "s3://far-research-internal/colbhben/gaze")
            self.assertEqual(cfg.upload_mode, "awscli")
            self.assertEqual(cfg.access_mode, "file_mount")
            self.assertEqual(cfg.unprocessed_uri("aea", "seq1", "video_main_rgb"), "s3://example-bucket/gaze/unprocessed/aea/seq1/video_main_rgb")
            self.assertEqual(cfg.processed_uri("default-10hz", "aea", "seq1"), "s3://example-bucket/gaze/processed/default-10hz/aea/seq1")
            self.assertEqual(cfg.processed_manifest_uri("default-10hz"), "s3://example-bucket/gaze/processed/default-10hz/manifest.jsonl")
            self.assertEqual(str(mount_path_for_uri(cfg, cfg.processed_uri("default-10hz", "aea", "seq1"))), "/mnt/s3/gaze/processed/default-10hz/aea/seq1")

    def test_serial_download_backup_dry_run_uses_unprocessed_prefix(self) -> None:
        cfg = S3Config(bucket_uri="s3://example-bucket/gaze")
        with tempfile.TemporaryDirectory() as tmp:
            report = serial_download_backup(
                REPO_ROOT,
                cfg,
                Path(tmp) / "raw",
                datasets={"aea"},
                modalities={"gaze"},
                sequences={"loc5_script4_seq6_rec1"},
                dry_run=True,
                workers=48,
            )
            self.assertEqual(len(report["operations"]), 1)
            operation = report["operations"][0]
            self.assertEqual(operation["fetch"]["workers"], 48)
            self.assertIn("s3://example-bucket/gaze/unprocessed/aea/loc5_script4_seq6_rec1/mps_eye_gaze/", operation["s3_uri"])
            self.assertEqual(operation["upload"]["transport"], "awscli")
            self.assertEqual(operation["upload"]["command"][0:3], ["aws", "s3", "cp"])
            self.assertTrue(any("s3://example-bucket/gaze/unprocessed/aea/loc5_script4_seq6_rec1/mps_eye_gaze/" in part for part in operation["upload"]["command"]))
            self.assertFalse(str(Path(tmp) / "raw").startswith("/nfs"))

    def test_backup_raw_selects_largest_local_batch_that_fits(self) -> None:
        cfg = S3Config(bucket_uri="s3://example-bucket/gaze")
        with tempfile.TemporaryDirectory() as tmp:
            report = serial_download_backup(
                REPO_ROOT,
                cfg,
                Path(tmp) / "raw",
                datasets={"aea"},
                modalities={"gaze", "annotation"},
                sequences={"loc5_script4_seq6_rec1"},
                max_download_bytes=1000,
                reserve_bytes=0,
                storage_fraction=1.0,
                dry_run=True,
            )
            self.assertEqual(report["selection"]["selected_assets"], 1)
            self.assertEqual(report["operations"][0]["asset"]["asset_key"], "annotations")
            skipped_keys = {asset["asset_key"] for asset in report["selection"]["skipped_assets"]}
            self.assertIn("mps_eye_gaze", skipped_keys)

    def test_serial_process_backup_dry_run_pulls_unprocessed_partition(self) -> None:
        cfg = S3Config(bucket_uri="s3://example-bucket/gaze")
        report = serial_process_backup(cfg, [("toy", "ep1")], dry_run=True)
        download = report["operations"][0]["download"]
        self.assertEqual(download["transport"], "file_mount")
        self.assertEqual(download["operation"], "sync")
        self.assertEqual(download["source"], "s3://example-bucket/gaze/unprocessed/toy/ep1")
        self.assertEqual(download["source_path"], "/nfs/gaze/unprocessed/toy/ep1")

    def test_pull_manifest_creation_and_pull_dry_run(self) -> None:
        cfg = S3Config(bucket_uri="s3://example-bucket/gaze")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            split = tmp_path / "default.json"
            split.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": ["toy:ep1"],
                            "holdout": ["toy:ep2"],
                        }
                    }
                )
            )
            pull_manifest = tmp_path / "default.s3.json"
            manifest = create_s3_pull_manifest(cfg, split, pull_manifest)
            self.assertEqual(len(manifest["episodes"]), 2)
            self.assertEqual(manifest["episodes"][0]["s3_uri"], "s3://example-bucket/gaze/processed/default-10hz/toy/ep1")
            self.assertEqual(manifest["episodes"][0]["nfs_path"], "/nfs/gaze/processed/default-10hz/toy/ep1")

            report = pull_processed_from_manifest(cfg, pull_manifest, tmp_path / "canonical", split="train", dry_run=True)
            self.assertEqual(len(report["operations"]), 1)
            self.assertEqual(report["operations"][0]["source"], "s3://example-bucket/gaze/processed/default-10hz/toy/ep1")
            self.assertEqual(report["operations"][0]["source_path"], "/nfs/gaze/processed/default-10hz/toy/ep1")


if __name__ == "__main__":
    unittest.main()
