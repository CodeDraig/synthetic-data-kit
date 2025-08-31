import os
import json
import tempfile
from pathlib import Path
import importlib

import pytest


def _reload_server_with_root(root: str):
    os.environ["SDK_BROWSE_ROOT"] = root
    import synthetic_data_kit.server.app as server_app
    return importlib.reload(server_app)


@pytest.mark.functional
def test_api_browse_lists_and_filters(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        # Create structure
        Path(tmp, "docs").mkdir()
        Path(tmp, "a.txt").write_text("A")
        Path(tmp, "b.pdf").write_text("B")
        Path(tmp, ".hidden.txt").write_text("H")

        server_app = _reload_server_with_root(tmp)
        client = server_app.app.test_client()

        # Root listing
        resp = client.get("/api/browse")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [e["name"] for e in data["entries"]]
        # Hidden file excluded
        assert "docs" in names and "a.txt" in names and "b.pdf" in names and ".hidden.txt" not in names

        # Directories only
        resp = client.get("/api/browse?dirsOnly=true")
        assert resp.status_code == 200
        data = resp.get_json()
        assert all(e["type"] == "dir" for e in data["entries"])
        assert any(e["name"] == "docs" for e in data["entries"])

        # Extension filter
        resp = client.get("/api/browse?ext=.txt")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [e["name"] for e in data["entries"]]
        assert "a.txt" in names and "b.pdf" not in names

        # Navigate into subdirectory
        resp = client.get("/api/browse?path=docs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["cwd"] == "docs"
        assert data["entries"] == []


@pytest.mark.functional
def test_api_browse_forbidden_and_not_directory(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "file.txt").write_text("X")
        server_app = _reload_server_with_root(tmp)
        client = server_app.app.test_client()

        # Path traversal attempt
        resp = client.get("/api/browse?path=..")
        assert resp.status_code == 403
        assert resp.get_json().get("error") == "Forbidden"

        # Not a directory
        resp = client.get("/api/browse?path=file.txt")
        assert resp.status_code == 400
        assert resp.get_json().get("error") == "Not a directory"
