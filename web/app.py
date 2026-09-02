#!/usr/bin/env python3
"""
Kannada OCR & Autocorrect Pipeline — Flask Web Application
Provides interactive Web Dashboard and REST APIs for document OCR, live correction, and export.
"""

import os
import sys
import uuid
import json
import time
import queue
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    process_document,
    process_text_input,
    init_pipeline
)
from pipeline.ocr import (
    is_tesseract_available,
    get_available_languages,
    DEFAULT_PSM,
    DEFAULT_OEM,
    SUPPORTED_LANGUAGES
)
from pipeline.correction import (
    get_dictionary, get_word_list, ENGINES, ENGINE_RULE, engine_status, preload_engine
)
from pipeline.ingestion import SUPPORTED_IMAGE_EXTENSIONS, inspect_pdf, is_pdf_file

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250MB limit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'web', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'web', 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.pdf'} | SUPPORTED_IMAGE_EXTENSIONS

# Session store for streaming jobs
_SESSIONS = {}

# Default correction engine for the dashboard. 'hybrid' rather than 'rule'
# because it makes measurably fewer wrong corrections (6 vs 20 per 100
# benchmark lines) at the same error rate, which is what this project's corpus
# goal actually cares about. Viable as a default only because of the warmup
# below -- without it, the first request of every session would stall ~20s
# loading the model.
DEFAULT_ENGINE = os.environ.get('PRAGNA_WEB_ENGINE', 'hybrid')

# Warmup happens on a background thread so the server binds its port and serves
# the page immediately; the UI polls this state and enables the LM engines when
# it flips to 'ready'. If the model can't load (no torch, no GPU), the state
# records why and the dashboard falls back to the rule engine rather than
# offering something that will fail.
_WARMUP = {'engine': DEFAULT_ENGINE, 'state': 'pending', 'error': '', 'seconds': 0.0}


def _warm_engine():
    """Load the dictionary, n-gram model and (if needed) the LM, once at boot."""
    started = time.time()
    _WARMUP['state'] = 'loading'
    try:
        init_pipeline()
        preload_engine(DEFAULT_ENGINE)
        _WARMUP['state'] = 'ready'
    except Exception as e:
        # Not fatal: the rule engine needs none of this, so the dashboard stays
        # usable and simply reports the LM as unavailable.
        _WARMUP['state'] = 'failed'
        _WARMUP['error'] = str(e)
        print(f"[warmup] {DEFAULT_ENGINE} engine unavailable: {e}")
    finally:
        _WARMUP['seconds'] = round(time.time() - started, 1)


def start_warmup():
    if DEFAULT_ENGINE == ENGINE_RULE:
        _WARMUP.update({'state': 'ready', 'seconds': 0.0})
        threading.Thread(target=init_pipeline, daemon=True).start()
        return
    threading.Thread(target=_warm_engine, daemon=True).start()


def effective_default_engine() -> str:
    """The engine the UI should preselect, degraded if warmup failed."""
    return ENGINE_RULE if _WARMUP['state'] == 'failed' else DEFAULT_ENGINE


def is_allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


@app.before_request
def setup_pipeline_on_start():
    # Also covers being run under a WSGI server, where __main__ never executes
    # and start_warmup() would otherwise never fire.
    if _WARMUP['state'] == 'pending':
        start_warmup()
    init_pipeline()


@app.route('/')
def index():
    return render_template('index.html')


def requested_engine(value) -> str:
    """
    Validate a client-supplied engine name, falling back to the rule engine.

    Unknown names fall back rather than 400 so a stale browser tab can't break
    a long OCR job over a dropdown value; an engine that is *known but
    unavailable* still raises, because silently correcting with a different
    engine than the user picked would misreport what produced the text.
    """
    engine = (value or effective_default_engine()).strip()
    if engine not in ENGINES:
        return effective_default_engine()
    status = engine_status()[engine]
    if not status['available']:
        raise RuntimeError(f"The '{engine}' engine is unavailable. {status['reason']}")
    return engine


@app.route('/api/system-status', methods=['GET'])
def system_status():
    """Return backend status, installed OCR engines, dictionary stats, and engines."""
    tess_installed = is_tesseract_available()
    avail_langs = get_available_languages() if tess_installed else []

    return jsonify({
        'tesseract_available': tess_installed,
        'installed_languages': avail_langs,
        'supported_languages': SUPPORTED_LANGUAGES,
        'dictionary_words_count': len(get_word_list()),
        'max_upload_size_mb': 250,
        'engines': engine_status(),
        'default_engine': effective_default_engine(),
        'warmup': dict(_WARMUP)
    })


@app.route('/api/correct-text', methods=['POST'])
def api_correct_text():
    """Instant autocorrect for raw text."""
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    try:
        engine = requested_engine(data.get('engine'))
        result = process_text_input(text, engine=engine)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Correction failed: {e}'}), 500
    return jsonify(result)


import re


def safe_upload_filename(filename: str) -> str:
    """Safely sanitize uploaded filename preserving extension and Kannada/Unicode characters."""
    ext = os.path.splitext(filename)[1].lower()
    base = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r'[\/\\:\*\?"<>\|\x00]', '_', base).strip('._ ')
    if not cleaned:
        cleaned = "document"
    return f"{cleaned}{ext}"


@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    """
    Step 1: Fast file upload endpoint with progress tracking support.
    Saves file and inspects basic metadata (e.g. page count).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if not file.filename or not is_allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    session_id = uuid.uuid4().hex
    safe_name = safe_upload_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_name}")
    file.save(upload_path)

    file_size_mb = round(os.path.getsize(upload_path) / (1024 * 1024), 2)
    total_pages = 1

    if is_pdf_file(upload_path):
        try:
            info = inspect_pdf(upload_path)
            total_pages = info['page_count']
        except Exception as e:
            print(f"Error inspecting PDF: {e}")
            total_pages = 1

    _SESSIONS[session_id] = {
        'upload_path': upload_path,
        'filename': safe_name,
        'file_size_mb': file_size_mb,
        'total_pages': total_pages,
        'created_at': time.time()
    }

    return jsonify({
        'success': True,
        'session_id': session_id,
        'filename': safe_name,
        'file_size_mb': file_size_mb,
        'total_pages': total_pages
    })



@app.route('/api/process-stream/<session_id>', methods=['GET'])
def api_process_stream(session_id: str):
    """
    Step 2: Server-Sent Events (SSE) stream for real-time page-by-page progress.
    """
    session_data = _SESSIONS.get(session_id)
    if not session_data:
        return jsonify({'error': 'Invalid or expired upload session.'}), 404

    lang = request.args.get('lang', 'kan')
    dpi = int(request.args.get('dpi', 400))
    psm = int(request.args.get('psm', DEFAULT_PSM))
    oem = int(request.args.get('oem', DEFAULT_OEM))
    save_images = request.args.get('save_images', 'false').lower() == 'true'
    try:
        engine = requested_engine(request.args.get('engine'))
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503

    upload_path = session_data['upload_path']
    output_dir = os.path.join(PROCESSED_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)

    event_q = queue.Queue()

    def progress_callback(event):
        event_q.put(event)

    def worker():
        try:
            res = process_document(
                input_path=upload_path,
                lang=lang,
                dpi=dpi,
                psm=psm,
                oem=oem,
                engine=engine,
                output_dir=output_dir,
                save_pdf=True,
                save_images=save_images,
                progress_callback=progress_callback
            )
            event_q.put({
                'stage': 'complete',
                'percent': 100,
                'message': 'Document processing complete!',
                'payload': {
                    'session_id': session_id,
                    'filename': session_data['filename'],
                    'result': res['report'],
                    'raw_text': res['raw_text'],
                    'corrected_text': res['corrected_text'],
                    'total_pages': res['total_pages'],
                    'total_corrections': res['total_corrections'],
                    'latency_seconds': res['latency_seconds'],
                    'engine': engine,
                    'download_urls': {
                        'pdf': f'/api/download/{session_id}/pdf',
                        'txt': f'/api/download/{session_id}/txt',
                        'json': f'/api/download/{session_id}/json'
                    }
                }
            })
        except Exception as e:
            event_q.put({
                'stage': 'error',
                'error': str(e),
                'message': f'Error: {str(e)}'
            })

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def generate_events():
        while True:
            try:
                event = event_q.get(timeout=60)
                data_str = json.dumps(event, ensure_ascii=False)
                yield f"data: {data_str}\n\n"

                if event.get('stage') in ('complete', 'error'):
                    break
            except queue.Empty:
                # Keep-alive heartbeat
                yield f": heartbeat\n\n"

    return Response(generate_events(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    })


@app.route('/api/process-document', methods=['POST'])
def api_process_document():
    """Direct processing fallback endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if not file.filename or not is_allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    lang = request.form.get('lang', 'kan')
    dpi = int(request.form.get('dpi', 400))
    psm = int(request.form.get('psm', DEFAULT_PSM))
    oem = int(request.form.get('oem', DEFAULT_OEM))
    save_images = request.form.get('save_images', 'false').lower() == 'true'
    try:
        engine = requested_engine(request.form.get('engine'))
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503

    session_id = uuid.uuid4().hex
    safe_name = safe_upload_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_name}")
    file.save(upload_path)

    output_dir = os.path.join(PROCESSED_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = process_document(
            input_path=upload_path,
            lang=lang,
            dpi=dpi,
            psm=psm,
            oem=oem,
            engine=engine,
            output_dir=output_dir,
            save_pdf=True,
            save_images=save_images
        )

        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': safe_name,
            'result': result['report'],
            'raw_text': result['raw_text'],
            'corrected_text': result['corrected_text'],
            'total_pages': result['total_pages'],
            'total_corrections': result['total_corrections'],
            'latency_seconds': result['latency_seconds'],
            'engine': engine,
            'download_urls': {
                'pdf': f'/api/download/{session_id}/pdf',
                'txt': f'/api/download/{session_id}/txt',
                'json': f'/api/download/{session_id}/json'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<session_id>/<file_type>', methods=['GET'])
def download_output(session_id: str, file_type: str):
    """Download generated output file by session ID."""
    out_dir = os.path.join(PROCESSED_FOLDER, secure_filename(session_id))
    if not os.path.exists(out_dir):
        return jsonify({'error': 'File expired or not found.'}), 404

    files = os.listdir(out_dir)

    if file_type == 'pdf':
        target = next((f for f in files if f.endswith('_corrected.pdf')), None)
        mimetype = 'application/pdf'
    elif file_type == 'txt':
        target = next((f for f in files if f.endswith('_corrected.txt')), None)
        mimetype = 'text/plain'
    elif file_type == 'json':
        target = next((f for f in files if f.endswith('_report.json')), None)
        mimetype = 'application/json'
    else:
        return jsonify({'error': 'Invalid file type requested.'}), 400

    if not target:
        return jsonify({'error': f'{file_type.upper()} file not found for this session.'}), 404

    target_path = os.path.join(out_dir, target)
    return send_file(target_path, mimetype=mimetype, as_attachment=True, download_name=target)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5010))
    print(f"Starting Kannada OCR & Autocorrect Web App on http://127.0.0.1:{port}")
    print(f"Default correction engine: {DEFAULT_ENGINE} (warming in background)")
    start_warmup()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
