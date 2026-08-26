let currentEngineMode = 'hybrid';
let currentAiModel = 'qwen2.5:3b';

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initEngineSelector();
    initLiveAutocorrect();
    initDocumentUpload();
    initSystemStatus();
});

/* ── Engine Mode Selector ── */
function initEngineSelector() {
    const pills = document.querySelectorAll('.engine-pill');
    const modelSelect = document.getElementById('aiModelSelect');

    pills.forEach(pill => {
        pill.addEventListener('click', () => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentEngineMode = pill.getAttribute('data-mode') || 'hybrid';
            
            // Re-trigger live text correction if text is present
            const liveInput = document.getElementById('liveRawInput');
            if (liveInput && liveInput.value.trim()) {
                liveInput.dispatchEvent(new Event('input'));
            }
        });
    });

    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            currentAiModel = modelSelect.value;
            const liveInput = document.getElementById('liveRawInput');
            if (liveInput && liveInput.value.trim()) {
                liveInput.dispatchEvent(new Event('input'));
            }
        });
    }
}

/* ── Top-Level Shared Utilities ── */

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function escapeRegExp(string) {
    if (!string) return '';
    return String(string).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getDiffClass(type) {
    if (type === 'hybrid') return 'diff-yellow';
    if (type === 'ocr_repair') return 'diff-blue';
    return 'diff-green';
}

function getTypeBadge(type) {
    if (type === 'hybrid') return '<span class="tag-badge yellow">Dual (OCR + Word)</span>';
    if (type === 'ocr_repair') return '<span class="tag-badge blue">OCR Glyph Repair</span>';
    return '<span class="tag-badge green">Word Correction</span>';
}

function highlightCorrectedHtml(text, corrections) {
    if (!text) return '';
    if (!corrections || !Array.isArray(corrections) || corrections.length === 0) {
        return escapeHtml(text);
    }

    try {
        const map = new Map();
        corrections.forEach(c => {
            if (c && c.correction && c.original && c.correction !== c.original) {
                if (!map.has(c.correction)) {
                    map.set(c.correction, c);
                }
            }
        });

        if (map.size === 0) return escapeHtml(text);

        const sortedKeys = Array.from(map.keys()).sort((a, b) => b.length - a.length);
        const pattern = sortedKeys.map(k => escapeRegExp(k)).join('|');
        const masterRegex = new RegExp(`(${pattern})`, 'g');

        let result = '';
        let lastIndex = 0;
        let match;

        while ((match = masterRegex.exec(text)) !== null) {
            const matchIndex = match.index;
            const matchedText = match[0];

            result += escapeHtml(text.substring(lastIndex, matchIndex));

            const c = map.get(matchedText);
            const diffClass = getDiffClass(c ? c.type : 'ocr_repair');
            const orig = escapeHtml(c ? c.original : matchedText);

            result += `<mark class="apple-diff-pill ${diffClass}" title="Original OCR: ${orig}">${escapeHtml(matchedText)}</mark>`;

            lastIndex = matchIndex + matchedText.length;
        }

        result += escapeHtml(text.substring(lastIndex));
        return result;

    } catch (e) {
        console.error("Highlight rendering error:", e);
        return escapeHtml(text);
    }
}


/* ── Apple Segmented Pill Tab Switcher ── */
function initTabs() {
    const tabButtons = document.querySelectorAll('.segment-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.add('active');
        });
    });
}


/* ── Tab 1: Live Interactive Text Autocorrect ── */
function initLiveAutocorrect() {
    const rawInput = document.getElementById('liveRawInput');
    const correctedDisplay = document.getElementById('liveCorrectedDisplay');
    const statWords = document.getElementById('liveStatWords');
    const statFixes = document.getElementById('liveStatFixes');
    const statAccuracy = document.getElementById('liveStatAccuracy');
    const statLatency = document.getElementById('liveStatLatency');
    const tableBody = document.getElementById('liveCorrectionsTableBody');
    const copyBtn = document.getElementById('liveCopyBtn');
    const clearBtn = document.getElementById('liveClearBtn');

    let debounceTimer = null;
    let latestCorrectedText = '';

    const sampleText = "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬ ವ್ಯಕ್ತಿಯ ಜಿವನದಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತದೆ. ಇದು ಸಮಾಜದ ಶಿಕಷಣ ಅಭಿವೃದ್ಧಿಗೆ ಕಾರಣವಾಗುತದೆ.";

    const loadSampleBtn = document.getElementById('liveLoadSampleBtn');
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', () => {
            if (rawInput) {
                rawInput.value = sampleText;
                triggerCorrection();
            }
        });
    }

    if (rawInput) {
        rawInput.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(triggerCorrection, 300);
        });
    }

    async function triggerCorrection() {
        const text = (rawInput.value || '').trim();
        if (!text) {
            correctedDisplay.innerHTML = '<span style="color: var(--text-tertiary);">Corrected text will appear here in real time...</span>';
            statWords.textContent = '0';
            statFixes.textContent = '0';
            statAccuracy.textContent = '100%';
            statLatency.textContent = '0.00s';
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-tertiary); padding: 30px;">Type or paste Kannada text above to see live token corrections.</td></tr>';
            latestCorrectedText = '';
            return;
        }

        try {
            const resp = await fetch('/api/correct-text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    engine_mode: currentEngineMode,
                    ai_model: currentAiModel
                })
            });
            const data = await resp.json();

            if (resp.ok) {
                renderLiveResults(data);
            }
        } catch (err) {
            console.error("Autocorrect error:", err);
        }
    }

    function renderLiveResults(data) {
        latestCorrectedText = data.corrected || '';
        statWords.textContent = data.total_words || 0;
        statFixes.textContent = data.total_corrections || 0;
        statAccuracy.textContent = `${data.accuracy_rate || 100}%`;
        statLatency.textContent = `${data.latency_seconds || 0}s`;

        correctedDisplay.innerHTML = highlightCorrectedHtml(data.corrected, data.corrections);

        if (data.corrections && data.corrections.length > 0) {
            tableBody.innerHTML = data.corrections.map((c, idx) => `
                <tr>
                    <td style="font-family: var(--font-mono); color: var(--text-tertiary); font-size: 13px;">${idx + 1}</td>
                    <td class="tag-red-strike">${escapeHtml(c.original)}</td>
                    <td class="tag-green-bold">${escapeHtml(c.correction)}</td>
                    <td>${getTypeBadge(c.type)}</td>
                    <td style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 13px;">${c.edit_distance}</td>
                </tr>
            `).join('');
        } else {
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--apple-emerald); padding: 24px; font-weight: 500;">✓ Perfect match! All Kannada words are valid & clean.</td></tr>';
        }
    }

    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            if (latestCorrectedText) {
                navigator.clipboard.writeText(latestCorrectedText);
                copyBtn.textContent = 'Copied ✓';
                setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (rawInput) {
                rawInput.value = '';
                triggerCorrection();
            }
        });
    }
}


/* ── Tab 2: Document Upload & Real-Time Pipeline Stream ── */
function initDocumentUpload() {
    const dropzone = document.getElementById('uploadDropzone') || document.getElementById('docDropzone');
    const fileInput = document.getElementById('docFileInput');
    const btnBrowseFiles = document.getElementById('btnBrowseFiles');
    const selectedFileCard = document.getElementById('selectedFileCard');
    const selectedFileNameText = document.getElementById('selectedFileNameText');
    const selectedFileSizeText = document.getElementById('selectedFileSizeText');
    const fileIconBadge = document.getElementById('fileIconBadge');
    const btnRemoveFile = document.getElementById('btnRemoveFile');

    const langSelect = document.getElementById('docLangSelect') || document.getElementById('ocrLangSelect');
    const dpiSelect = document.getElementById('docDpiSelect') || document.getElementById('dpiSelect');
    const processBtn = document.getElementById('docProcessBtn') || document.getElementById('btnProcessDoc');
    
    const progressCard = document.getElementById('docProgressCard');
    const progressStageBadge = document.getElementById('docProgressStageBadge') || document.getElementById('docProgressStagePill');
    const progressPercent = document.getElementById('docProgressPercent');
    const progressBarFill = document.getElementById('docProgressBarFill');
    const progressStatusText = document.getElementById('docProgressStatusText');
    const progressMetaText = document.getElementById('docProgressMetaText');
    
    const docResultsCard = document.getElementById('docResultsCard');
    const docRawDisplay = document.getElementById('docRawDisplay');
    const docCorrectedDisplay = document.getElementById('docCorrectedDisplay');
    const docStatPages = document.getElementById('docStatPages');
    const docStatFixes = document.getElementById('docStatFixes');
    const docStatTime = document.getElementById('docStatTime');

    const btnCopyDocText = document.getElementById('btnCopyDocText');
    const btnDownloadPdf = document.getElementById('btnDownloadPdf');
    const btnDownloadTxt = document.getElementById('btnDownloadTxt');
    const btnDownloadJson = document.getElementById('btnDownloadJson');
    const docCorrectionsTableBody = document.getElementById('docCorrectionsTableBody');

    let selectedFile = null;
    let activeEventSource = null;
    let latestDocCorrectedText = '';

    if (btnCopyDocText) {
        btnCopyDocText.addEventListener('click', () => {
            if (latestDocCorrectedText) {
                navigator.clipboard.writeText(latestDocCorrectedText);
                btnCopyDocText.textContent = 'Copied ✓';
                setTimeout(() => { btnCopyDocText.textContent = 'Copy Text'; }, 2000);
            }
        });
    }

    // Trigger file chooser on dropzone click or Browse button
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', (e) => {
            if (e.target !== btnBrowseFiles) {
                fileInput.click();
            }
        });

        if (btnBrowseFiles) {
            btnBrowseFiles.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.click();
            });
        }

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileSelected(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files[0]) {
                handleFileSelected(fileInput.files[0]);
            }
        });
    }

    if (btnRemoveFile) {
        btnRemoveFile.addEventListener('click', (e) => {
            e.stopPropagation();
            clearSelectedFile();
            if (fileInput) fileInput.click();
        });
    }

    function clearSelectedFile() {
        selectedFile = null;
        if (fileInput) fileInput.value = '';
        if (selectedFileCard) selectedFileCard.style.display = 'none';
        if (dropzone) {
            dropzone.style.borderColor = '';
            dropzone.style.background = '';
        }
        if (processBtn) {
            processBtn.disabled = true;
            processBtn.textContent = 'Run Ingest & Correction';
        }
    }

    function handleFileSelected(file) {
        selectedFile = file;
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        const ext = (file.name.split('.').pop() || '').toLowerCase();

        let icon = '📄';
        if (ext === 'pdf') icon = '📑';
        else if (['jpg', 'jpeg', 'png', 'webp', 'tiff', 'tif', 'bmp'].includes(ext)) icon = '🖼️';

        if (fileIconBadge) fileIconBadge.textContent = icon;
        if (selectedFileNameText) selectedFileNameText.textContent = file.name;
        if (selectedFileSizeText) selectedFileSizeText.textContent = `${sizeMb} MB • Ready for Ingestion`;

        if (selectedFileCard) {
            selectedFileCard.style.display = 'flex';
        }

        if (dropzone) {
            dropzone.style.borderColor = 'var(--apple-blue)';
            dropzone.style.background = 'rgba(0, 113, 227, 0.04)';
        }

        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = `Process "${file.name.length > 20 ? file.name.substring(0, 18) + '...' : file.name}"`;
        }
    }

    if (processBtn) {
        processBtn.addEventListener('click', () => {
            if (!selectedFile) {
                if (fileInput) fileInput.click();
                return;
            }

            processBtn.disabled = true;
            processBtn.textContent = 'Processing Document...';
            if (docResultsCard) docResultsCard.style.display = 'none';
            if (progressCard) {
                progressCard.style.display = 'block';
                progressCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            if (progressStageBadge) progressStageBadge.textContent = 'Uploading';
            if (progressPercent) progressPercent.textContent = '0%';
            if (progressBarFill) progressBarFill.style.width = '0%';
            if (progressStatusText) progressStatusText.textContent = `Uploading ${selectedFile.name}...`;
            if (progressMetaText) progressMetaText.textContent = `0 / ${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`;

            // Phase 1: Fast upload
            const xhr = new XMLHttpRequest();
            const formData = new FormData();
            formData.append('file', selectedFile);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const upPct = Math.round((e.loaded / e.total) * 100);
                    const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
                    const totalMb = (e.total / (1024 * 1024)).toFixed(1);

                    if (progressPercent) progressPercent.textContent = `${Math.min(upPct, 99)}%`;
                    if (progressBarFill) progressBarFill.style.width = `${Math.min(upPct * 0.20, 20)}%`;
                    if (progressStatusText) progressStatusText.textContent = `Uploading document to server (${upPct}%)...`;
                    if (progressMetaText) progressMetaText.textContent = `${loadedMb} / ${totalMb} MB`;
                }
            };

            xhr.onload = () => {
                if (xhr.status === 200) {
                    try {
                        const uploadRes = JSON.parse(xhr.responseText);
                        if (uploadRes.success && uploadRes.session_id) {
                            startProcessingStream(uploadRes.session_id);
                        } else {
                            fallbackDirectProcessing();
                        }
                    } catch (e) {
                        fallbackDirectProcessing();
                    }
                } else {
                    fallbackDirectProcessing();
                }
            };

            xhr.onerror = () => {
                fallbackDirectProcessing();
            };

            xhr.open('POST', '/api/upload', true);
            xhr.send(formData);
        });
    }

    function fallbackDirectProcessing() {
        if (progressStatusText) progressStatusText.textContent = 'Processing document directly via OCR engine...';
        if (progressBarFill) progressBarFill.style.width = '50%';
        if (progressPercent) progressPercent.textContent = '50%';

        const lang = langSelect ? langSelect.value : 'kan+eng';
        const dpi = dpiSelect ? dpiSelect.value : 300;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('lang', lang);
        formData.append('dpi', dpi);
        formData.append('engine_mode', currentEngineMode);
        if (currentAiModel) formData.append('ai_model', currentAiModel);

        fetch('/api/process-document', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(`Processing failed: ${data.error}`);
                resetProgress();
            } else {
                if (progressPercent) progressPercent.textContent = '100%';
                if (progressBarFill) progressBarFill.style.width = '100%';
                setTimeout(() => {
                    resetProgress();
                    renderDocumentResults(data);
                }, 300);
            }
        })
        .catch(err => {
            alert(`Document processing error: ${err.message || err}`);
            resetProgress();
        });
    }

    function startProcessingStream(sessionId) {
        const lang = langSelect ? langSelect.value : 'kan+eng';
        const dpi = dpiSelect ? dpiSelect.value : 300;

        if (progressStageBadge) progressStageBadge.textContent = 'Processing';
        if (progressStatusText) progressStatusText.textContent = 'Connecting to OCR engine pipeline...';
        if (progressBarFill) progressBarFill.style.width = '20%';

        if (activeEventSource) {
            activeEventSource.close();
        }

        let streamReceivedAny = false;
        const streamUrl = `/api/process-stream/${sessionId}?lang=${encodeURIComponent(lang)}&dpi=${encodeURIComponent(dpi)}&engine_mode=${encodeURIComponent(currentEngineMode)}&ai_model=${encodeURIComponent(currentAiModel)}`;
        activeEventSource = new EventSource(streamUrl);

        activeEventSource.onmessage = (e) => {
            streamReceivedAny = true;
            try {
                const event = JSON.parse(e.data);

                if (event.stage === 'inspecting') {
                    if (progressStageBadge) progressStageBadge.textContent = 'Inspecting';
                    if (progressBarFill) progressBarFill.style.width = '25%';
                    if (progressPercent) progressPercent.textContent = '25%';
                    if (progressStatusText) progressStatusText.textContent = event.message || 'Analyzing document layout...';
                } else if (event.stage === 'rasterizing') {
                    if (progressStageBadge) progressStageBadge.textContent = 'Rasterizing';
                    if (progressBarFill) progressBarFill.style.width = '35%';
                    if (progressPercent) progressPercent.textContent = '35%';
                    if (progressStatusText) progressStatusText.textContent = event.message || 'Rendering high-DPI page sheets...';
                } else if (event.stage === 'ocr') {
                    if (progressStageBadge) progressStageBadge.textContent = 'OCR Engine';
                    const pct = Math.max(35, Math.min(event.percent || 50, 85));
                    if (progressBarFill) progressBarFill.style.width = `${pct}%`;
                    if (progressPercent) progressPercent.textContent = `${pct}%`;
                    if (progressStatusText) progressStatusText.textContent = event.message || `OCR Processing Page ${event.current_page || 1}...`;
                    if (progressMetaText) progressMetaText.textContent = `Page ${event.current_page || 1} of ${event.total_pages || 1}`;
                } else if (event.stage === 'extracting') {
                    if (progressStageBadge) progressStageBadge.textContent = 'Extracting';
                    if (progressBarFill) progressBarFill.style.width = '60%';
                    if (progressPercent) progressPercent.textContent = '60%';
                    if (progressStatusText) progressStatusText.textContent = event.message || 'Extracting digital vector layout...';
                } else if (event.stage === 'correcting') {
                    if (progressStageBadge) progressStageBadge.textContent = 'Correcting';
                    if (progressBarFill) progressBarFill.style.width = '88%';
                    if (progressPercent) progressPercent.textContent = '88%';
                    if (progressStatusText) progressStatusText.textContent = event.message || 'Applying Kannada autocorrect & morphology...';
                } else if (event.stage === 'exporting') {
                    if (progressStageBadge) progressStageBadge.textContent = 'Exporting';
                    if (progressBarFill) progressBarFill.style.width = '95%';
                    if (progressPercent) progressPercent.textContent = '95%';
                    if (progressStatusText) progressStatusText.textContent = event.message || 'Compiling output PDF & reports...';
                } else if (event.stage === 'finalizing') {
                    if (progressStageBadge) progressStageBadge.textContent = 'Finalizing';
                    if (progressBarFill) progressBarFill.style.width = '98%';
                    if (progressPercent) progressPercent.textContent = '98%';
                    if (progressStatusText) progressStatusText.textContent = event.message || 'Preparing report...';
                } else if (event.stage === 'complete' && event.payload) {
                    if (progressPercent) progressPercent.textContent = '100%';
                    if (progressBarFill) progressBarFill.style.width = '100%';
                    if (progressStatusText) progressStatusText.textContent = 'Complete ✓';
                    activeEventSource.close();
                    setTimeout(() => {
                        resetProgress();
                        renderDocumentResults(event.payload);
                    }, 400);
                } else if (event.stage === 'error') {
                    activeEventSource.close();
                    fallbackDirectProcessing();
                }
            } catch (err) {
                console.error("SSE parse error:", err);
            }
        };

        activeEventSource.onerror = () => {
            if (!streamReceivedAny) {
                activeEventSource.close();
                fallbackDirectProcessing();
            }
        };
    }

    function resetProgress() {
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = 'Run Ingest & Correction';
        }
        if (progressCard) progressCard.style.display = 'none';
        if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
        }
    }

    function renderDocumentResults(data) {
        if (!data) return;
        latestDocCorrectedText = data.corrected_text || '';
        if (docResultsCard) docResultsCard.style.display = 'block';

        if (docStatPages) docStatPages.textContent = data.total_pages ?? (data.result ? data.result.total_pages : 0);
        if (docStatFixes) docStatFixes.textContent = data.total_corrections ?? (data.result ? data.result.total_corrections : 0);
        if (docStatTime) docStatTime.textContent = `${data.latency_seconds || 0}s`;

        if (docRawDisplay) docRawDisplay.textContent = data.raw_text || '(No text extracted)';

        // 3-color Highlight (Blue = OCR repair, Green = Word correction, Yellow = Both)
        const corrections = (data.result && data.result.corrections_summary) ? data.result.corrections_summary : [];
        if (docCorrectedDisplay) {
            docCorrectedDisplay.innerHTML = highlightCorrectedHtml(data.corrected_text, corrections) || '(No text extracted)';
        }

        if (btnDownloadPdf && data.download_urls) btnDownloadPdf.href = data.download_urls.pdf;
        if (btnDownloadTxt && data.download_urls) btnDownloadTxt.href = data.download_urls.txt;
        if (btnDownloadJson && data.download_urls) btnDownloadJson.href = data.download_urls.json;

        // Render detailed corrections with Classification Badges
        if (docCorrectionsTableBody) {
            if (corrections.length > 0) {
                docCorrectionsTableBody.innerHTML = corrections.map((c, idx) => `
                    <tr>
                        <td style="font-family: var(--font-mono); color: var(--text-tertiary); font-size: 13px;">${idx + 1}</td>
                        <td class="tag-red-strike">${escapeHtml(c.original)}</td>
                        <td class="tag-green-bold">${escapeHtml(c.correction)}</td>
                        <td>${getTypeBadge(c.type)}</td>
                        <td style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 13px;">${c.edit_distance}</td>
                    </tr>
                `).join('');
            } else {
                docCorrectionsTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--apple-emerald); padding: 24px; font-weight: 500;">✓ Document text is completely clean! No corrections were required.</td></tr>';
            }
        }

        if (docResultsCard) {
            docResultsCard.scrollIntoView({ behavior: 'smooth' });
        }
    }
}


/* ── Tab 3: System Status ── */
async function initSystemStatus() {
    try {
        const resp = await fetch('/api/system-status');
        const data = await resp.json();

        const badge = document.getElementById('headerStatusBadge');
        if (badge) {
            badge.innerHTML = data.tesseract_available
                ? `<span class="nav-status-dot"></span> Tesseract Active`
                : `<span style="color: var(--apple-cyan);">●</span> Morphological Engine Online`;
        }

        const tessStatusEl = document.getElementById('diagTessStatus');
        const dictCountEl = document.getElementById('diagDictCount');
        const langTagsEl = document.getElementById('diagLangTags');
        const ollamaStatusEl = document.getElementById('diagOllamaStatus');
        const activeAiModelEl = document.getElementById('diagActiveAiModel');
        const aiStatusPill = document.getElementById('aiStatusPill');
        const aiStatusText = document.getElementById('aiStatusText');
        const modelSelect = document.getElementById('aiModelSelect');

        if (tessStatusEl) {
            tessStatusEl.textContent = data.tesseract_available ? 'Active' : 'Standby / Digital Mode';
            tessStatusEl.className = data.tesseract_available ? 'apple-stat-value emerald' : 'apple-stat-value cyan';
        }

        if (dictCountEl) {
            dictCountEl.textContent = data.dictionary_words_count.toLocaleString();
        }

        // Ollama AI Diagnostics
        if (ollamaStatusEl) {
            ollamaStatusEl.textContent = data.ollama_available ? 'Online (Active)' : 'Offline / Standby';
            ollamaStatusEl.className = data.ollama_available ? 'apple-stat-value emerald' : 'apple-stat-value';
        }

        if (activeAiModelEl) {
            activeAiModelEl.textContent = data.default_ai_model || 'qwen2.5:3b';
        }

        if (aiStatusPill && aiStatusText) {
            if (data.ollama_available) {
                aiStatusPill.className = 'ai-status-pill online';
                aiStatusText.textContent = `Ollama Online (${data.ollama_models && data.ollama_models.length ? data.ollama_models[0] : 'Ready'})`;
            } else {
                aiStatusPill.className = 'ai-status-pill offline';
                aiStatusText.textContent = 'Ollama Standby';
            }
        }

        // Populate Model Select Options
        if (modelSelect && data.ollama_models && data.ollama_models.length > 0) {
            modelSelect.innerHTML = data.ollama_models.map(m => `
                <option value="${escapeHtml(m)}">${escapeHtml(m)}</option>
            `).join('');
            currentAiModel = data.ollama_models[0];
        }

        if (langTagsEl && data.installed_languages && data.installed_languages.length > 0) {
            langTagsEl.innerHTML = data.installed_languages.map(l => `
                <span class="apple-btn apple-btn-secondary apple-btn-sm">${l}</span>
            `).join(' ');
        }
    } catch (e) {
        console.error("Diagnostic error:", e);
    }
}
