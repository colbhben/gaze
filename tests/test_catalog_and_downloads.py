from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gaze.datasets import filter_assets, load_catalog, parse_datasets_md
from gaze.download import estimate_downloads, fetch_assets, plan_downloads, write_download_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    def test_datasets_md_and_manifests_are_parsed(self) -> None:
        docs = parse_datasets_md(REPO_ROOT / "DATASETS.md")
        self.assertIn("holoassist", docs)
        self.assertIn("aea", docs)
        self.assertIn("hot3d", docs)
        self.assertIn("nymeria", docs)
        self.assertTrue(docs["aea"].manifest_paths)

        catalog = load_catalog(REPO_ROOT)
        assets = catalog.manifest_assets("aea")
        self.assertTrue(any(asset.modality == "video" for asset in assets))
        self.assertTrue(any(asset.modality == "gaze" for asset in assets))
        self.assertTrue(any(asset.modality == "annotation" for asset in assets))

    def test_download_plan_filters_and_estimates(self) -> None:
        catalog = load_catalog(REPO_ROOT)
        assets = plan_downloads(catalog, datasets={"aea"}, modalities={"video"}, sequences={"loc5_script4_seq6_rec1"})
        self.assertGreaterEqual(len(assets), 1)
        self.assertTrue(all(asset.dataset == "aea" for asset in assets))
        self.assertTrue(all(asset.modality == "video" for asset in assets))
        estimate = estimate_downloads(assets)
        self.assertEqual(estimate[0]["assets"], len(assets))
        self.assertGreater(estimate[0]["bytes"], 0)

    def test_manifest_output_and_dry_run_fetch_do_not_download(self) -> None:
        catalog = load_catalog(REPO_ROOT)
        assets = filter_assets(catalog.manifest_assets("hot3d"), modalities={"gaze"}, sequences={"P0001_10a27bf7"})
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "download_manifest.json"
            write_download_manifest(assets, manifest)
            self.assertTrue(manifest.exists())
            fetched = fetch_assets(assets, Path(tmp) / "raw", dry_run=True)
            self.assertEqual(len(fetched), 1)
            self.assertTrue(fetched[0]["dry_run"])
            self.assertEqual(fetched[0]["workers"], 1)
            self.assertFalse((Path(tmp) / "raw").exists())

            threaded = fetch_assets(assets, Path(tmp) / "raw", dry_run=True, workers=8)
            self.assertEqual(threaded[0]["workers"], 8)


if __name__ == "__main__":
    unittest.main()
