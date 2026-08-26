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
    SUPPORTED_LANGUAGES
)
from pipeline.correction import get_dictionary, get_word_list
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


def is_allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


@app.before_request
def setup_pipeline_on_start():
    init_pipeline()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/system-status', methods=['GET'])
def system_status():
    """Return backend status, installed OCR engines, and dictionary stats."""
    tess_installed = is_tesseract_available()
    avail_langs = get_available_languages() if tess_installed else []
    
    return jsonify({
        'tesseract_available': tess_installed,
        'installed_languages': avail_langs,
        'supported_languages': SUPPORTED_LANGUAGES,
        'dictionary_words_count': len(get_word_list()),
        'max_upload_size_mb': 250
    })


@app.route('/api/correct-text', methods=['POST'])
def api_correct_text():
    """Instant autocorrect for raw text."""
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    result = process_text_input(text)
    return jsonify(result)


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
    safe_name = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_name}")
    file.save(upload_path)

    file_size_mb = round(os.path.getsize(upload_path) / (1024 * 1024), 2)
    total_pages = 1

    if is_pdf_file(upload_path):
        try:
            info = inspect_pdf(upload_path)
            total_pages = info['page_count']
        except Exception:
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

    lang = request.args.get('lang', 'kan+eng')
    dpi = int(request.args.get('dpi', 300))
    save_images = request.args.get('save_images', 'false').lower() == 'true'

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

    lang = request.form.get('lang', 'kan+eng')
    dpi = int(request.form.get('dpi', 300))
    save_images = request.form.get('save_images', 'false').lower() == 'true'

    session_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_name}")
    file.save(upload_path)

    output_dir = os.path.join(PROCESSED_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = process_document(
            input_path=upload_path,
            lang=lang,
            dpi=dpi,
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
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Kannada OCR & Autocorrect Web App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
