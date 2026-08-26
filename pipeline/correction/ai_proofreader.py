"""
Local Indic AI Proofreading Module powered by Ollama
Provides neural contextual post-OCR proofreading for Kannada text
using local Small Language Models (e.g. Qwen 2.5 3B, Llama 3.2 3B, Gemma 2 2B).
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
DEFAULT_MODEL = os.environ.get('KANNADA_AI_MODEL', 'qwen2.5:3b')

SYSTEM_PROMPT = """You are an expert Indic OCR post-processing engine specializing in Kannada (ಕನ್ನಡ).
Your goal is to correct optical character recognition (OCR) errors in Kannada text.

STRICT RULES:
1. Fix OCR glyph corruptions, broken subscript ligatures (Ottu), incorrect vowel signs (Matras), and misrecognized consonants.
2. Fix broken words split by erroneous whitespace (e.g. 'ಸಾಧ್ಯ ವಾಗಬೇಕಿದೆ' -> 'ಸಾಧ್ಯವಾಗಬೇಕಿದೆ').
3. Clean Unicode glitches (e.g. Kannada digit zero '೦' mistakenly scanned instead of Anusvara 'ಂ').
4. STRICT CONTENT FIDELITY: Do NOT summarize, do NOT rewrite, do NOT modernize vocabulary, and do NOT hallucinate. Keep the exact sentence structure and author's original words.
5. Return ONLY the corrected Kannada text. Do not provide greetings, markdown code blocks, intros, or explanations.
"""


def is_ollama_running() -> bool:
    """Check if Ollama service is active and reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", headers={'User-Agent': 'PRAGNA_OCR'})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_models() -> List[str]:
    """Retrieve list of locally installed models in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", headers={'User-Agent': 'PRAGNA_OCR'})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = [m.get('name', '') for m in data.get('models', []) if m.get('name')]
            return models
    except Exception:
        return []


def query_ollama_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.1,
    timeout_sec: float = 120.0
) -> str:
    """
    Send prompt to local Ollama chat/generate API and return text response.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 1024,
            "num_thread": 8
        }
    }

    req_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={'Content-Type': 'application/json', 'User-Agent': 'PRAGNA_OCR'},
        method='POST'
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        res_json = json.loads(resp.read().decode('utf-8'))
        raw_response = res_json.get('response', '').strip()

        # Clean any markdown block formatting if the model wrapped output in ```
        raw_response = re.sub(r'^```(?:kannada|text)?\n', '', raw_response, flags=re.IGNORECASE)
        raw_response = re.sub(r'\n```$', '', raw_response)
        return raw_response.strip()


def compute_token_diffs(raw_text: str, corrected_text: str) -> List[Dict[str, Any]]:
    """
    Compute structured word-level diffs between raw OCR text and AI corrected text.
    """
    import difflib
    from pipeline.correction.edit_distance import weighted_edit_distance

    raw_tokens = re.findall(r'\S+', raw_text)
    corr_tokens = re.findall(r'\S+', corrected_text)

    matcher = difflib.SequenceMatcher(None, raw_tokens, corr_tokens)
    corrections = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            orig = " ".join(raw_tokens[i1:i2])
            corr = " ".join(corr_tokens[j1:j2])

            if orig and corr and orig != corr:
                dist = round(weighted_edit_distance(orig, corr), 2)
                corrections.append({
                    'original': orig,
                    'correction': corr,
                    'edit_distance': dist,
                    'type': 'word_correction' if dist <= 2.0 else 'ocr_repair'
                })

    return corrections


def proofread_kannada_ai(
    text: str,
    model: Optional[str] = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    High-level entry point to proofread Kannada OCR text using local AI.
    Returns: {
        'original': str,
        'corrected': str,
        'corrections': List[Dict],
        'total_words': int,
        'total_corrections': int,
        'accuracy_rate': float,
        'latency_seconds': float,
        'model_used': str,
        'ai_available': bool
    }
    """
    start_time = time.time()
    text = (text or '').strip()

    if not text:
        return {
            'original': '',
            'corrected': '',
            'corrections': [],
            'total_words': 0,
            'total_corrections': 0,
            'accuracy_rate': 100.0,
            'latency_seconds': 0.0,
            'model_used': 'none',
            'ai_available': False
        }

    installed = get_installed_models()
    chosen_model = model or (installed[0] if installed else DEFAULT_MODEL)

    if not is_ollama_running() or not installed:
        # Graceful fallback to algorithmic corrector
        from pipeline.correction.corrector import correct_text
        res = correct_text(text)
        res['ai_available'] = False
        res['model_used'] = 'fallback:algorithmic'
        return res

    try:
        # Smart line/sentence chunking to keep CPU inference fast (<3s per chunk)
        lines = text.split('\n')
        chunks = []
        curr = []
        curr_words = 0

        for line in lines:
            w_count = len(line.split())
            if curr_words + w_count > 35 and curr:
                chunks.append('\n'.join(curr))
                curr = [line]
                curr_words = w_count
            else:
                curr.append(line)
                curr_words += w_count
        if curr:
            chunks.append('\n'.join(curr))

        corrected_chunks = []
        for ch in chunks:
            ch_strip = ch.strip()
            if not ch_strip:
                corrected_chunks.append('')
                continue

            user_prompt = f"Correct the OCR errors in the following Kannada text:\n\n{ch_strip}"
            corr_ch = query_ollama_chat(
                prompt=user_prompt,
                model=chosen_model,
                temperature=temperature
            )
            corrected_chunks.append(corr_ch if corr_ch else ch_strip)

        final_corrected = "\n".join(corrected_chunks)
        diffs = compute_token_diffs(text, final_corrected)
        words = re.findall(r'\S+', text)
        total_words = len(words)
        total_corrections = len(diffs)
        accuracy_rate = round(max(0.0, min(100.0, 100.0 - (total_corrections / max(1, total_words) * 100))), 1)
        latency = round(time.time() - start_time, 2)

        return {
            'original': text,
            'corrected': final_corrected,
            'corrections': diffs,
            'total_words': total_words,
            'total_corrections': total_corrections,
            'accuracy_rate': accuracy_rate,
            'latency_seconds': latency,
            'model_used': chosen_model,
            'ai_available': True
        }

    except Exception as e:
        print(f"[AI Proofreader] Error during Ollama query: {e}. Falling back to algorithmic engine.")
        from pipeline.correction.corrector import correct_text
        res = correct_text(text)
        res['ai_available'] = False
        res['model_used'] = f'fallback:algorithmic (error: {str(e)[:30]})'
        return res
