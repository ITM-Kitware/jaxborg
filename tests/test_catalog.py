import json

import tests.catalog as catalog


class TestCatalogStatus:
    def test_update_l3_coverage_with_existing_metadata_keys(self, tmp_path, monkeypatch):
        status_path = tmp_path / "catalog_status.json"
        status_path.write_text(
            json.dumps(
                {
                    "1": "passing",
                    "l3_coverage": {"seeds": 5, "steps": 500, "clean": True},
                }
            )
        )
        monkeypatch.setattr(catalog, "CATALOG_STATUS_PATH", status_path)

        catalog.update_l3_coverage(seeds=20, steps=500, clean=True)

        saved = json.loads(status_path.read_text())
        assert saved["1"] == "passing"
        assert saved["l3_coverage"] == {"seeds": 20, "steps": 500, "clean": True}
