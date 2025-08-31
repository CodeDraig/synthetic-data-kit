from io import BytesIO
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest


def _reload_server():
    import synthetic_data_kit.server.app as server_app
    return importlib.reload(server_app)


@pytest.mark.functional
def test_ingest_multi_file_upload_redirects_to_files(monkeypatch):
    server_app = _reload_server()
    server_app.app.config['WTF_CSRF_ENABLED'] = False
    client = server_app.app.test_client()

    with patch("synthetic_data_kit.server.app.ingest_process_file") as mock_proc:
        mock_proc.side_effect = [
            str(server_app.DEFAULT_OUTPUT_DIR / "out1.txt"),
            str(server_app.DEFAULT_OUTPUT_DIR / "out2.txt"),
        ]
        data = {
            'input_type': 'file',
            'upload_file': [
                (BytesIO(b"hello"), 'a.txt'),
                (BytesIO(b"world"), 'b.txt'),
            ],
            'submit': 'Parse Document'
        }
        resp = client.post('/ingest', data=data, content_type='multipart/form-data')

        assert resp.status_code == 302
        assert resp.headers['Location'].endswith('/files')
        assert mock_proc.call_count == 2


@pytest.mark.functional
def test_ingest_unsupported_extension_redirects_back(monkeypatch):
    server_app = _reload_server()
    server_app.app.config['WTF_CSRF_ENABLED'] = False
    client = server_app.app.test_client()

    with patch("synthetic_data_kit.server.app.ingest_process_file") as mock_proc:
        data = {
            'input_type': 'file',
            'upload_file': [(BytesIO(b"bad"), 'bad.xyz')],
            'submit': 'Parse Document'
        }
        resp = client.post('/ingest', data=data, content_type='multipart/form-data')

        # No processing occurred; redirect back to /ingest
        assert resp.status_code == 302
        assert resp.headers['Location'].endswith('/ingest')
        mock_proc.assert_not_called()


@pytest.mark.functional
def test_ingest_no_files_renders_form(monkeypatch):
    server_app = _reload_server()
    server_app.app.config['WTF_CSRF_ENABLED'] = False
    client = server_app.app.test_client()

    # Post with input_type=file but no upload_file field
    data = {'input_type': 'file', 'submit': 'Parse Document'}
    resp = client.post('/ingest', data=data)

    assert resp.status_code == 200
    assert b"Please upload at least one file" in resp.data


@pytest.mark.functional
def test_ingest_url_input_redirects_to_view(monkeypatch):
    server_app = _reload_server()
    server_app.app.config['WTF_CSRF_ENABLED'] = False
    client = server_app.app.test_client()

    out_path = str(server_app.DEFAULT_OUTPUT_DIR / "result.txt")
    with patch("synthetic_data_kit.server.app.ingest_process_file", return_value=out_path) as mock_proc:
        data = {
            'input_type': 'url',
            'input_path': 'https://example.com/article',
            'output_name': 'result',
            'submit': 'Parse Document'
        }
        resp = client.post('/ingest', data=data)

        assert resp.status_code == 302
        # Should redirect to view_file
        assert '/view/' in resp.headers['Location']
        mock_proc.assert_called_once()
