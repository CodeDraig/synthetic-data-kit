import importlib
from unittest.mock import patch

import pytest


def _reload_server_with_config(cfg):
    # Patch load_config during module import so module-level config is our cfg
    with patch("synthetic_data_kit.utils.config.load_config", return_value=cfg):
        import synthetic_data_kit.server.app as server_app
        return importlib.reload(server_app)


@pytest.mark.functional
def test_settings_get_renders(monkeypatch):
    cfg = {
        'paths': {
            'input': 'data/input',
            'output': {
                'parsed': 'data/parsed',
                'generated': 'data/generated',
                'curated': 'data/curated',
                'final': 'data/final',
                'default': 'data/output'
            }
        },
        'llm': {'provider': 'vllm'},
        'vllm': {'api_base': 'http://localhost:8000/v1', 'model': 'llama'},
        'generation': {'temperature': 0.7, 'top_p': 0.95, 'max_tokens': 4096}
    }
    server_app = _reload_server_with_config(cfg)
    client = server_app.app.test_client()

    resp = client.get('/settings')
    assert resp.status_code == 200
    # Basic smoke check on template content
    assert b"Settings" in resp.data


@pytest.mark.functional
def test_settings_post_updates_config(monkeypatch):
    cfg = {
        'paths': {
            'input': 'data/input',
            'output': {
                'parsed': 'data/parsed',
                'generated': 'data/generated',
                'curated': 'data/curated',
                'final': 'data/final',
                'default': 'data/output'
            }
        },
        'llm': {'provider': 'vllm'},
        'vllm': {'api_base': 'http://localhost:8000/v1', 'model': 'llama', 'port': 8000, 'max_retries': 3, 'retry_delay': 1.0},
        'api-endpoint': {'api_base': None, 'model': 'gpt-4o', 'max_retries': 3, 'retry_delay': 1.0},
        'ollama': {'api_base': 'http://localhost:11434', 'model': 'llama3', 'max_retries': 3, 'retry_delay': 1.0, 'sleep_time': 0.1},
        'generation': {'temperature': 0.7, 'top_p': 0.95, 'max_tokens': 4096, 'chunk_size': 4000, 'overlap': 200}
    }
    server_app = _reload_server_with_config(cfg)
    server_app.app.config['WTF_CSRF_ENABLED'] = False
    client = server_app.app.test_client()

    form_data = {
        'provider': 'ollama',
        'ollama_model': 'llama3.1',
        'paths_input_default': '/tmp/input',
        'paths_output_default': '/tmp/output',
        'gen_temperature': '0.2',
        'gen_top_p': '0.9',
        'gen_max_tokens': '1234',
        'submit': 'Save Settings'
    }

    resp = client.post('/settings', data=form_data, follow_redirects=True)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Settings updated" in body

    # Config should be updated in-memory on the module
    assert server_app.get_llm_provider(server_app.config) == 'ollama'
    gen_conf = server_app.get_generation_config(server_app.config)
    assert gen_conf.get('temperature') == 0.2
    assert gen_conf.get('top_p') == 0.9
    assert gen_conf.get('max_tokens') == 1234

    # Paths should reflect overrides
    in_path = server_app.get_path_config(server_app.config, 'input')
    out_path = server_app.get_path_config(server_app.config, 'output')
    assert in_path == '/tmp/input'
    assert out_path == '/tmp/output'
