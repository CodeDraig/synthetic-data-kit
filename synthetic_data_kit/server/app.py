"""
Flask application for the Synthetic Data Kit web interface.
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import flask
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort, flash
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, IntegerField, SelectField, FileField, SubmitField, FloatField, PasswordField
from wtforms.validators import DataRequired, Optional as OptionalValidator

from synthetic_data_kit.utils.config import (
    load_config,
    get_llm_provider,
    get_path_config,
    get_vllm_config,
    get_openai_config,
    get_ollama_config,
    get_generation_config,
    merge_configs,
)
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs
from synthetic_data_kit.core.ingest import process_file as ingest_process_file
from werkzeug.utils import secure_filename
from synthetic_data_kit.utils.directory_processor import INGEST_EXTENSIONS, CREATE_EXTENSIONS

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Set default paths
DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "output"
DEFAULT_GENERATED_DIR = DEFAULT_DATA_DIR / "generated"

# Create directories if they don't exist
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Load SDK config
config = load_config()

# Derived UI constants (accept strings and labels) and safe browse root
# Keep accept strings in sync with supported extensions from directory_processor
ACCEPT_INGEST = ",".join(INGEST_EXTENSIONS)
ACCEPT_UPLOAD = ACCEPT_INGEST
ACCEPT_CREATE = ",".join(CREATE_EXTENSIONS)

# Root directory for server-side file browsing (defaults to project root containing 'data/')
BROWSE_ROOT = os.environ.get("SDK_BROWSE_ROOT", str(Path(DEFAULT_DATA_DIR.parent).resolve()))
BROWSE_ROOT_PATH = Path(BROWSE_ROOT).resolve()

def _safe_join(base: Path, *paths: str) -> Path:
    """Join paths ensuring the result stays within base (prevents path traversal)."""
    target = (base.joinpath(*paths)).resolve()
    if not str(target).startswith(str(base)):
        raise PermissionError("Path traversal outside browse root")
    return target

# Forms
class CreateForm(FlaskForm):
    """Form for creating content from text"""
    input_file = StringField('Input File Path', validators=[DataRequired()])
    content_type = SelectField('Content Type', choices=[
        ('qa', 'Question-Answer Pairs'), 
        ('summary', 'Summary'), 
        ('cot', 'Chain of Thought'), 
        ('cot-enhance', 'CoT Enhancement')
    ], default='qa')
    num_pairs = IntegerField('Number of QA Pairs', default=10)
    model = StringField('Model Name (optional)')
    api_base = StringField('API Base URL (optional)')
    submit = SubmitField('Generate Content')
    
class IngestForm(FlaskForm):
    """Form for ingesting documents"""
    input_type = SelectField('Input Type', choices=[
        ('file', 'Upload File'),
        ('url', 'URL'),
        ('path', 'Local Path')
    ], default='file')
    upload_file = FileField('Upload Document')
    input_path = StringField('File Path or URL')
    output_name = StringField('Output Filename (optional)')
    submit = SubmitField('Parse Document')

class CurateForm(FlaskForm):
    """Form for curating QA pairs"""
    input_file = StringField('Input JSON File Path', validators=[DataRequired()])
    num_pairs = IntegerField('Number of QA Pairs to Keep', default=0)
    model = StringField('Model Name (optional)')
    api_base = StringField('API Base URL (optional)')
    submit = SubmitField('Curate QA Pairs')

class UploadForm(FlaskForm):
    """Form for uploading files"""
    file = FileField('Upload File', validators=[DataRequired()])
    submit = SubmitField('Upload')

class SettingsForm(FlaskForm):
    """Form for live configuration overrides"""
    # Provider selection
    provider = SelectField('LLM Provider', choices=[('vllm', 'vllm'), ('api-endpoint', 'api-endpoint'), ('ollama', 'ollama')])

    # VLLM
    vllm_api_base = StringField('VLLM API Base', validators=[OptionalValidator()])
    vllm_model = StringField('VLLM Model', validators=[OptionalValidator()])
    vllm_port = IntegerField('VLLM Port', validators=[OptionalValidator()])

    # API Endpoint
    api_ep_api_base = StringField('API Endpoint Base URL', validators=[OptionalValidator()])
    api_ep_model = StringField('API Endpoint Model', validators=[OptionalValidator()])
    api_ep_api_key = PasswordField('API Endpoint API Key', validators=[OptionalValidator()])

    # Ollama
    ollama_api_base = StringField('Ollama API Base', validators=[OptionalValidator()])
    ollama_model = StringField('Ollama Model', validators=[OptionalValidator()])

    # Paths
    paths_input_default = StringField('Input Directory (default)', validators=[OptionalValidator()])
    paths_output_default = StringField('Output Directory (default)', validators=[OptionalValidator()])

    # Generation
    gen_temperature = FloatField('Generation Temperature', validators=[OptionalValidator()])
    gen_top_p = FloatField('Top P', validators=[OptionalValidator()])
    gen_max_tokens = IntegerField('Max Tokens', validators=[OptionalValidator()])

    submit = SubmitField('Save Settings')

# Routes
@app.route('/')
def index():
    """Main index page"""
    provider = get_llm_provider(config)
    return render_template('index.html', provider=provider)

@app.route('/create', methods=['GET', 'POST'])
def create():
    """Create content from text"""
    form = CreateForm()
    provider = get_llm_provider(config)
    
    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            content_type = form.content_type.data
            num_pairs = form.num_pairs.data
            # Prefer form values; otherwise pull from current in-memory settings
            if provider == 'vllm':
                prov_conf = get_vllm_config(config)
            elif provider == 'api-endpoint':
                prov_conf = get_openai_config(config)
            else:
                prov_conf = get_ollama_config(config)

            model = form.model.data or prov_conf.get('model')
            api_base = form.api_base.data or prov_conf.get('api_base')
            
            output_path = process_file(
                file_path=input_file,
                output_dir=str(DEFAULT_GENERATED_DIR),
                content_type=content_type,
                num_pairs=num_pairs,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True
            )
            
            content_type_labels = {
                'qa': 'QA pairs',
                'summary': 'summary',
                'cot': 'Chain of Thought examples',
                'cot-enhance': 'CoT enhanced conversation'
            }
            content_label = content_type_labels.get(content_type, content_type)
            
            flash(f'Successfully generated {content_label}! Output saved to: {output_path}', 'success')
            return redirect(url_for('view_file', file_path=str(Path(output_path).relative_to(DEFAULT_DATA_DIR.parent))))
            
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    # Get the list of available input files
    # - .txt from configured input directory (uploads)
    # - .txt and .lance datasets from output (parsed/lance from ingest)
    input_files = []
    try:
        input_dir_str = get_path_config(config, 'input')
    except Exception:
        input_dir_str = 'data/input'

    input_dir_path = Path(input_dir_str)
    if not input_dir_path.is_absolute():
        input_dir_path = Path(DEFAULT_DATA_DIR.parent) / input_dir_path

    txt_files: List[Path] = []
    if input_dir_path.exists():
        txt_files.extend([f for f in input_dir_path.glob('*.txt')])
    if DEFAULT_OUTPUT_DIR.exists():
        txt_files.extend([f for f in DEFAULT_OUTPUT_DIR.glob('*.txt')])
        lance_dirs = [f for f in DEFAULT_OUTPUT_DIR.glob('*.lance') if f.is_dir()]
    else:
        lance_dirs = []

    combined = txt_files + lance_dirs
    input_files = [str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in sorted(combined, key=lambda p: p.name.lower())]
    
    return render_template(
        'create.html',
        form=form,
        provider=provider,
        input_files=input_files,
        create_accept=ACCEPT_CREATE,
        browse_root=str(BROWSE_ROOT_PATH),
    )

@app.route('/curate', methods=['GET', 'POST'])
def curate():
    """Curate QA pairs interface"""
    form = CurateForm()
    provider = get_llm_provider(config)
    
    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            num_pairs = form.num_pairs.data
            # Prefer form values; otherwise pull from current in-memory settings
            if provider == 'vllm':
                prov_conf = get_vllm_config(config)
            elif provider == 'api-endpoint':
                prov_conf = get_openai_config(config)
            else:
                prov_conf = get_ollama_config(config)

            model = form.model.data or prov_conf.get('model')
            api_base = form.api_base.data or prov_conf.get('api_base')
            
            # Create output path
            filename = Path(input_file).stem
            output_file = f"{filename}_curated.json"
            output_path = str(Path(DEFAULT_GENERATED_DIR) / output_file)
            
            result_path = curate_qa_pairs(
                input_path=input_file,
                output_path=output_path,
                provider=provider,
                api_base=api_base, 
                model=model,
                config_path=None,  # Use default config
                verbose=True
            )
            
            flash(f'Successfully curated QA pairs! Output saved to: {result_path}', 'success')
            return redirect(url_for('view_file', file_path=str(Path(result_path).relative_to(DEFAULT_DATA_DIR.parent))))
            
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    # Get the list of available JSON files
    json_files = []
    if DEFAULT_GENERATED_DIR.exists():
        json_files = [str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_GENERATED_DIR.glob('*.json')]
    
    return render_template('curate.html', form=form, provider=provider, json_files=json_files)

@app.route('/files')
def files():
    """File browser"""
    # List input directory files and generated files
    input_files: List[str] = []
    generated_files: List[str] = []

    # Resolve configured input directory
    try:
        input_dir_str = get_path_config(config, 'input')
    except Exception:
        input_dir_str = 'data/input'
    input_dir_path = Path(input_dir_str)
    if not input_dir_path.is_absolute():
        input_dir_path = Path(DEFAULT_DATA_DIR.parent) / input_dir_path

    if input_dir_path.exists() and input_dir_path.is_dir():
        input_files = [str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in input_dir_path.glob('*.*')]

    if DEFAULT_GENERATED_DIR.exists():
        generated_files = [str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_GENERATED_DIR.glob('*.*')]

    return render_template('files.html', input_files=input_files, generated_files=generated_files)

@app.route('/view/<path:file_path>')
def view_file(file_path):
    """View a file's contents"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)
    
    if not full_path.exists():
        flash(f'File not found: {file_path}', 'danger')
        return redirect(url_for('files'))
    
    file_content = None
    file_type = "text"
    lance_columns: List[str] = []
    lance_rows: List[Dict[str, Any]] = []
    
    # Handle directories (e.g., Lance dataset directories)
    if full_path.is_dir():
        # Treat Lance datasets as special previewable directory types
        if full_path.suffix.lower() == '.lance':
            try:
                # Lazy import to avoid importing 'lance' unless needed
                try:
                    from synthetic_data_kit.utils.lance_utils import load_lance_dataset
                except Exception as ie:
                    file_content = f"[Lance dataset] Unable to import Lance utilities: {str(ie)}"
                    file_type = "text"
                    return render_template('view_file.html', 
                                          file_path=file_path, 
                                          file_type=file_type, 
                                          content=file_content,
                                          is_qa_pairs=False,
                                          is_cot_examples=False,
                                          has_conversations=False,
                                          has_summary=False,
                                          lance_columns=lance_columns,
                                          lance_rows=lance_rows)

                ds = load_lance_dataset(str(full_path))
                if ds is not None:
                    try:
                        table = ds.to_table(limit=50)
                    except Exception:
                        # Fallback: attempt to scan to a table if direct to_table fails
                        table = ds.scanner().limit(50).to_table()
                    lance_columns = list(table.column_names)
                    lance_rows = table.to_pylist()
                    file_type = "lance"
                else:
                    file_content = f"[Lance dataset] Dataset not found at {full_path}"
                    file_type = "text"
            except Exception as e:
                file_content = f"[Lance dataset] Unable to read dataset: {str(e)}"
                file_type = "text"
        else:
            # Generic directory handling: list entries for display as text
            try:
                entries = sorted(p.name for p in full_path.iterdir())
                file_content = "Directory contents:\n" + "\n".join(entries)
                file_type = "text"
            except Exception as e:
                file_content = f"[Directory] Unable to list contents: {str(e)}"
                file_type = "text"
        
        return render_template('view_file.html', 
                              file_path=file_path, 
                              file_type=file_type, 
                              content=file_content,
                              is_qa_pairs=False,
                              is_cot_examples=False,
                              has_conversations=False,
                              has_summary=False,
                              lance_columns=lance_columns,
                              lance_rows=lance_rows)
    
    if full_path.suffix.lower() == '.json':
        try:
            with open(full_path, 'r') as f:
                file_content = json.load(f)
            file_type = "json"
            
            # Detect specific JSON formats
            is_qa_pairs = 'qa_pairs' in file_content
            is_cot_examples = 'cot_examples' in file_content
            has_conversations = 'conversations' in file_content
            has_summary = 'summary' in file_content
            
        except Exception as e:
            # If JSON parsing fails, treat as text
            with open(full_path, 'r') as f:
                file_content = f.read()
            file_type = "text"
            is_qa_pairs = False
            is_cot_examples = False
            has_conversations = False
            has_summary = False
    else:
        # Read as text
        with open(full_path, 'r') as f:
            file_content = f.read()
        file_type = "text"
        is_qa_pairs = False
        is_cot_examples = False
        has_conversations = False
        has_summary = False
    
    return render_template('view_file.html', 
                          file_path=file_path, 
                          file_type=file_type, 
                          content=file_content,
                          is_qa_pairs=is_qa_pairs,
                          is_cot_examples=is_cot_examples,
                          has_conversations=has_conversations,
                          has_summary=has_summary,
                          lance_columns=lance_columns,
                          lance_rows=lance_rows)

@app.route('/ingest', methods=['GET', 'POST'])
def ingest():
    """Ingest and parse documents"""
    form = IngestForm()
    # Example URLs/labels used by the template in multiple code paths
    examples = {
        "PDF": "path/to/document.pdf",
        "YouTube": "https://www.youtube.com/watch?v=example",
        "Web Page": "https://example.com/article",
        "Word Document": "path/to/document.docx",
        "PowerPoint": "path/to/presentation.pptx",
        "Text File": "path/to/document.txt"
    }
    
    if form.validate_on_submit():
        try:
            input_type = form.input_type.data
            output_name = form.output_name.data or None
            
            # Get default output directory for parsed files
            output_dir = str(DEFAULT_OUTPUT_DIR)
            
            if input_type == 'file':
                # Handle single or multi-file upload
                files = request.files.getlist('upload_file')
                files = [f for f in files if getattr(f, 'filename', '')]
                if not files:
                    flash('Please upload at least one file', 'warning')
                    return render_template('ingest.html', form=form, examples=examples, ingest_accept=ACCEPT_INGEST, browse_root=str(BROWSE_ROOT_PATH))

                outputs = []
                temp_paths = []
                for idx, f in enumerate(files, start=1):
                    safe_name = secure_filename(f.filename)
                    ext = Path(safe_name).suffix.lower()
                    if ext not in INGEST_EXTENSIONS:
                        flash(f"Skipping unsupported file type: {safe_name}", 'warning')
                        continue

                    # Determine output base name
                    base_name = Path(safe_name).stem
                    this_output_name = output_name if (output_name and len(files) == 1) else base_name

                    # Save to a temporary path in output dir
                    temp_path = DEFAULT_OUTPUT_DIR / f"temp_{this_output_name}{ext}"
                    f.save(temp_path)
                    temp_paths.append(temp_path)

                    # Process the file
                    try:
                        result_path = ingest_process_file(
                            file_path=str(temp_path),
                            output_dir=output_dir,
                            output_name=this_output_name,
                            config=config
                        )
                        outputs.append(result_path)
                    except Exception as e:
                        flash(f"Failed to parse {safe_name}: {e}", 'danger')
                
                # Clean up temp files
                for tp in temp_paths:
                    try:
                        if tp.exists():
                            tp.unlink()
                    except Exception:
                        pass

                if not outputs:
                    return redirect(url_for('ingest'))

                if len(outputs) == 1:
                    flash(f'Successfully parsed document! Output saved to: {outputs[0]}', 'success')
                    return redirect(url_for('view_file', file_path=str(Path(outputs[0]).relative_to(DEFAULT_DATA_DIR.parent))))
                else:
                    flash(f'Successfully parsed {len(outputs)} documents.', 'success')
                    return redirect(url_for('files'))
            else:
                # URL or local path
                input_path = form.input_path.data
                if not input_path:
                    flash('Please enter a valid path or URL', 'warning')
                    return render_template('ingest.html', form=form, examples=examples, ingest_accept=ACCEPT_INGEST, browse_root=str(BROWSE_ROOT_PATH))
            
            # Process the file or URL
            output_path = ingest_process_file(
                file_path=input_path,
                output_dir=output_dir,
                output_name=output_name,
                config=config
            )
            
            flash(f'Successfully parsed document! Output saved to: {output_path}', 'success')
            return redirect(url_for('view_file', file_path=str(Path(output_path).relative_to(DEFAULT_DATA_DIR.parent))))
            
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    return render_template('ingest.html', form=form, examples=examples, ingest_accept=ACCEPT_INGEST, browse_root=str(BROWSE_ROOT_PATH))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload a file to the data directory"""
    form = UploadForm()
    
    if form.validate_on_submit():
        f = form.file.data
        filename = secure_filename(f.filename)
        # Determine input directory from config (paths.input)
        try:
            input_dir_str = get_path_config(config, 'input')
        except Exception:
            input_dir_str = 'data/input'

        input_dir_path = Path(input_dir_str)
        # If relative, resolve relative to project root (parent of data/)
        if not input_dir_path.is_absolute():
            input_dir_path = Path(DEFAULT_DATA_DIR.parent) / input_dir_path

        # Ensure directory exists
        input_dir_path.mkdir(parents=True, exist_ok=True)

        filepath = input_dir_path / filename
        f.save(filepath)
        flash(f'File uploaded successfully to input: {filename}', 'success')
        return redirect(url_for('files'))
    
    return render_template('upload.html', form=form, upload_accept=ACCEPT_UPLOAD)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page for live configuration overrides."""
    global config
    form = SettingsForm()

    # Load current values
    current_provider = get_llm_provider(config)
    vllm_conf = get_vllm_config(config)
    api_conf = get_openai_config(config)
    ollama_conf = get_ollama_config(config)
    input_dir = get_path_config(config, 'input')
    output_dir = get_path_config(config, 'output')
    gen_conf = get_generation_config(config)

    if request.method == 'GET':
        form.provider.data = current_provider
        form.vllm_api_base.data = vllm_conf.get('api_base')
        form.vllm_model.data = vllm_conf.get('model')
        form.vllm_port.data = vllm_conf.get('port')
        form.api_ep_api_base.data = api_conf.get('api_base')
        form.api_ep_model.data = api_conf.get('model')
        form.ollama_api_base.data = ollama_conf.get('api_base')
        form.ollama_model.data = ollama_conf.get('model')
        form.paths_input_default.data = input_dir
        form.paths_output_default.data = output_dir
        form.gen_temperature.data = gen_conf.get('temperature')
        form.gen_top_p.data = gen_conf.get('top_p')
        form.gen_max_tokens.data = gen_conf.get('max_tokens')

    if form.validate_on_submit():
        overrides = {
            'llm': {
                'provider': form.provider.data or current_provider
            },
            'vllm': {
                'api_base': form.vllm_api_base.data or vllm_conf.get('api_base'),
                'model': form.vllm_model.data or vllm_conf.get('model'),
                'port': form.vllm_port.data or vllm_conf.get('port'),
                'max_retries': vllm_conf.get('max_retries'),
                'retry_delay': vllm_conf.get('retry_delay'),
            },
            'api-endpoint': {
                'api_base': form.api_ep_api_base.data or api_conf.get('api_base'),
                'model': form.api_ep_model.data or api_conf.get('model'),
                'api_key': form.api_ep_api_key.data or api_conf.get('api_key'),
                'max_retries': api_conf.get('max_retries'),
                'retry_delay': api_conf.get('retry_delay'),
            },
            'ollama': {
                'api_base': form.ollama_api_base.data or ollama_conf.get('api_base'),
                'model': form.ollama_model.data or ollama_conf.get('model'),
                'max_retries': ollama_conf.get('max_retries'),
                'retry_delay': ollama_conf.get('retry_delay'),
                'sleep_time': ollama_conf.get('sleep_time'),
            },
            'paths': {
                'input': form.paths_input_default.data or input_dir,
                'output': {
                    'default': form.paths_output_default.data or output_dir
                }
            },
            'generation': {
                'temperature': form.gen_temperature.data if form.gen_temperature.data is not None else gen_conf.get('temperature'),
                'top_p': form.gen_top_p.data if form.gen_top_p.data is not None else gen_conf.get('top_p'),
                'chunk_size': gen_conf.get('chunk_size'),
                'overlap': gen_conf.get('overlap'),
                'max_tokens': form.gen_max_tokens.data if form.gen_max_tokens.data is not None else gen_conf.get('max_tokens'),
            }
        }

        # Apply overrides in-memory for live effect
        config = merge_configs(config, overrides)

        # Sync API key to environment for immediate effect without persistence
        try:
            effective_key = config.get('api-endpoint', {}).get('api_key')
            if effective_key:
                os.environ['API_ENDPOINT_KEY'] = effective_key
            else:
                # Remove to allow fallback to config file or no key
                os.environ.pop('API_ENDPOINT_KEY', None)
        except Exception:
            pass

        flash('Settings updated. Changes applied immediately for this server session.', 'success')
        return redirect(url_for('settings'))

    return render_template('settings.html', form=form, browse_root=str(BROWSE_ROOT_PATH))

@app.route('/api/test_provider', methods=['POST'])
def api_test_provider():
    """Test connectivity to the selected provider using current or provided settings.
    Body JSON: { provider, api_base?, api_key? }
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    # Determine provider (payload overrides current config)
    try:
        provider = (data.get('provider') or get_llm_provider(config))
    except Exception:
        provider = 'vllm'

    def resp(success: bool, **extra):
        return jsonify({ 'success': success, **extra })

    timeout = 5
    try:
        if provider == 'vllm':
            vllm_conf = get_vllm_config(config)
            base = (data.get('api_base') or vllm_conf.get('api_base') or '').strip()
            if not base:
                return resp(False, provider=provider, error='Missing api_base')
            base = base.rstrip('/')
            url = f"{base}/models" if base.endswith('/v1') else f"{base}/v1/models"
            r = requests.get(url, timeout=timeout)
            return resp(r.ok, provider=provider, url=url, status=r.status_code)

        if provider == 'ollama':
            ollama_conf = get_ollama_config(config)
            base = (data.get('api_base') or ollama_conf.get('api_base') or '').strip()
            if not base:
                return resp(False, provider=provider, error='Missing api_base')
            base = base.rstrip('/')
            url = f"{base}/api/tags"
            r = requests.get(url, timeout=timeout)
            return resp(r.ok, provider=provider, url=url, status=r.status_code)

        if provider == 'api-endpoint':
            api_conf = get_openai_config(config)
            base = data.get('api_base', api_conf.get('api_base'))
            if not base:
                base = 'https://api.openai.com/v1'
            base = str(base).rstrip('/')
            url = f"{base}/models" if base.endswith('/v1') else f"{base}/v1/models"
            # Choose API key: payload > env > config
            env_key = os.environ.get('API_ENDPOINT_KEY')
            key = data.get('api_key') or env_key or api_conf.get('api_key')
            headers = { 'Authorization': f'Bearer {key}' } if key else {}
            r = requests.get(url, headers=headers, timeout=timeout)
            return resp(r.ok, provider=provider, url=url, status=r.status_code)

        return resp(False, provider=provider, error=f'Unknown provider: {provider}')
    except requests.exceptions.RequestException as e:
        return resp(False, provider=provider, error=str(e))
    except Exception as e:
        return resp(False, provider=provider, error=str(e))

@app.route('/api/qa_json/<path:file_path>')
def qa_json(file_path):
    """Return QA pairs as JSON for the JSON viewer"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)
    
    if not full_path.exists() or full_path.suffix.lower() != '.json':
        abort(404)
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        abort(500)
        
@app.route('/api/edit_item/<path:file_path>', methods=['POST'])
def edit_item(file_path):
    """Edit an item in a JSON file"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)
    
    if not full_path.exists() or full_path.suffix.lower() != '.json':
        return jsonify({"success": False, "message": "File not found or not a JSON file"}), 404
    
    try:
        # Get the request data
        data = request.json
        item_type = data.get('item_type')  # qa_pairs, cot_examples, conversations
        item_index = data.get('item_index')
        item_content = data.get('item_content')
        
        if not all([item_type, item_index is not None, item_content]):
            return jsonify({"success": False, "message": "Missing required parameters"}), 400
        
        # Read the file
        with open(full_path, 'r') as f:
            file_content = json.load(f)
        
        # Update the item
        if item_type == 'qa_pairs' and 'qa_pairs' in file_content:
            if 0 <= item_index < len(file_content['qa_pairs']):
                file_content['qa_pairs'][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == 'cot_examples' and 'cot_examples' in file_content:
            if 0 <= item_index < len(file_content['cot_examples']):
                file_content['cot_examples'][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == 'conversations' and 'conversations' in file_content:
            if 0 <= item_index < len(file_content['conversations']):
                file_content['conversations'][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        else:
            return jsonify({"success": False, "message": "Invalid item type"}), 400
        
        # Write back to the file
        with open(full_path, 'w') as f:
            json.dump(file_content, f, indent=2)
        
        return jsonify({"success": True, "message": "Item updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/browse')
def api_browse():
    """Browse server-side filesystem within a safe root.
    Query params:
      - path: relative path from browse root
      - dirsOnly: 'true' to list only directories
      - ext: comma-separated list of extensions to include (e.g., .txt,.json)
    """
    rel_path = request.args.get('path', '').strip()
    dirs_only = request.args.get('dirsOnly', 'false').lower() == 'true'
    ext_param = request.args.get('ext', '').strip()
    ext_list = [e if e.startswith('.') else f'.{e}' for e in ext_param.split(',') if e]
    ext_list = [e.lower() for e in ext_list]

    try:
        current = _safe_join(BROWSE_ROOT_PATH, rel_path)
    except Exception:
        return jsonify({"error": "Forbidden"}), 403

    if not current.exists() or not current.is_dir():
        return jsonify({"error": "Not a directory"}), 400

    entries = []
    for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        # Skip hidden files/dirs by default
        if child.name.startswith('.'):
            continue
        if child.is_file() and dirs_only:
            continue
        if child.is_file() and ext_list and child.suffix.lower() not in ext_list:
            continue
        try:
            st = child.stat()
        except Exception:
            continue
        rel_child = str(child.relative_to(BROWSE_ROOT_PATH))
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "relPath": rel_child,
            "absPath": str(child)
        })

    # Build breadcrumbs
    try:
        rel_current = str(current.relative_to(BROWSE_ROOT_PATH))
    except Exception:
        rel_current = ""
    breadcrumbs = []
    if rel_current and rel_current != ".":
        parts = rel_current.split(os.sep)
        accum = ""
        for part in parts:
            accum = os.path.join(accum, part) if accum else part
            breadcrumbs.append({"name": part, "relPath": accum})

    parent_rel = "" if current == BROWSE_ROOT_PATH else str(current.parent.relative_to(BROWSE_ROOT_PATH))

    return jsonify({
        "root": str(BROWSE_ROOT_PATH),
        "cwd": rel_current if rel_current != "." else "",
        "entries": entries,
        "breadcrumbs": breadcrumbs,
        "parent": parent_rel,
    })

@app.route('/api/delete_item/<path:file_path>', methods=['POST'])
def delete_item(file_path):
    """Delete an item from a JSON file"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)
    
    if not full_path.exists() or full_path.suffix.lower() != '.json':
        return jsonify({"success": False, "message": "File not found or not a JSON file"}), 404
    
    try:
        # Get the request data
        data = request.json
        item_type = data.get('item_type')  # qa_pairs, cot_examples, conversations
        item_index = data.get('item_index')
        
        if not all([item_type, item_index is not None]):
            return jsonify({"success": False, "message": "Missing required parameters"}), 400
        
        # Read the file
        with open(full_path, 'r') as f:
            file_content = json.load(f)
        
        # Delete the item
        if item_type == 'qa_pairs' and 'qa_pairs' in file_content:
            if 0 <= item_index < len(file_content['qa_pairs']):
                file_content['qa_pairs'].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == 'cot_examples' and 'cot_examples' in file_content:
            if 0 <= item_index < len(file_content['cot_examples']):
                file_content['cot_examples'].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == 'conversations' and 'conversations' in file_content:
            if 0 <= item_index < len(file_content['conversations']):
                file_content['conversations'].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        else:
            return jsonify({"success": False, "message": "Invalid item type"}), 400
        
        # Write back to the file
        with open(full_path, 'w') as f:
            json.dump(file_content, f, indent=2)
        
        return jsonify({"success": True, "message": "Item deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the Flask server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server(debug=True)
