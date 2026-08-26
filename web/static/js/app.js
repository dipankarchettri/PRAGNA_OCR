/**
 * Kannada OCR & Autocorrect — Apple-Grade Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initLiveAutocorrect();
    initDocumentUpload();
    initSystemStatus();
});

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

    // Sample button
    const loadSampleBtn = document.getElementById('liveLoadSampleBtn');
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', () => {
            rawInput.value = sampleText;
            triggerCorrection();
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
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-tertiary); padding: 30px;">Type or paste Kannada text above to see live token corrections.</td></tr>';
            latestCorrectedText = '';
            return;
        }

        try {
            const resp = await fetch('/api/correct-text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
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
        latestCorrectedText = data.corrected;
        statWords.textContent = data.total_words || 0;
        statFixes.textContent = data.total_corrections || 0;
        statAccuracy.textContent = `${data.accuracy_rate || 100}%`;
        statLatency.textContent = `${data.latency_seconds || 0}s`;

        // Highlight corrections in the output display
        let highlighted = escapeHtml(data.corrected);
        if (data.corrections && data.corrections.length > 0) {
            data.corrections.forEach(c => {
                const escapedCorr = escapeHtml(c.correction);
                const reg = new RegExp(`(${escapedCorr})`, 'g');
                highlighted = highlighted.replace(reg, `<mark class="apple-diff-pill" title="Original: ${escapeHtml(c.original)}">$1</mark>`);
            });
        }
        correctedDisplay.innerHTML = highlighted;

        // Render Table
        if (data.corrections && data.corrections.length > 0) {
            tableBody.innerHTML = data.corrections.map((c, idx) => `
                <tr>
                    <td style="font-family: var(--font-mono); color: var(--text-tertiary); font-size: 13px;">${idx + 1}</td>
                    <td class="tag-red-strike">${escapeHtml(c.original)}</td>
                    <td class="tag-green-bold">${escapeHtml(c.correction)}</td>
                    <td style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 13px;">${c.edit_distance}</td>
                </tr>
            `).join('');
        } else {
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--apple-emerald); padding: 24px; font-weight: 500;">✓ Perfect match! All Kannada words are valid & clean.</td></tr>';
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
            rawInput.value = '';
            triggerCorrection();
        });
    }
}

/* ── Tab 2: Document Upload & Real-Time Pipeline Stream ── */
function initDocumentUpload() {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('docFileInput');
    const langSelect = document.getElementById('docLangSelect');
    const dpiSelect = document.getElementById('docDpiSelect');
    const processBtn = document.getElementById('docProcessBtn');
    
    const progressCard = document.getElementById('docProgressCard');
    const progressStageBadge = document.getElementById('docProgressStageBadge');
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

    const btnDownloadPdf = document.getElementById('btnDownloadPdf');
    const btnDownloadTxt = document.getElementById('btnDownloadTxt');
    const btnDownloadJson = document.getElementById('btnDownloadJson');
    const docCorrectionsTableBody = document.getElementById('docCorrectionsTableBody');

    let selectedFile = null;
    let activeEventSource = null;

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileSelected(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files[0]) {
                handleFileSelected(fileInput.files[0]);
            }
        });
    }

    function handleFileSelected(file) {
        selectedFile = file;
        const nameEl = document.getElementById('selectedFileName');
        if (nameEl) {
            nameEl.textContent = `✓ Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
            nameEl.style.display = 'block';
        }
        if (dropzone) {
            dropzone.style.borderColor = 'var(--apple-blue)';
            dropzone.style.background = 'rgba(0, 113, 227, 0.05)';
        }
        processBtn.disabled = false;
        processBtn.textContent = 'Run Ingest & Correction';
    }

    if (processBtn) {
        processBtn.addEventListener('click', () => {
            if (!selectedFile) {
                if (fileInput) fileInput.click();
                return;
            }

            processBtn.disabled = true;
            processBtn.textContent = 'Processing...';
            docResultsCard.style.display = 'none';
            progressCard.style.display = 'block';

            progressStageBadge.textContent = 'Uploading';
            progressPercent.textContent = '0%';
            progressBarFill.style.width = '0%';
            progressStatusText.textContent = `Uploading ${selectedFile.name}...`;
            progressMetaText.textContent = `0 / ${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`;

            // Phase 1: Upload with byte progress tracking
            const xhr = new XMLHttpRequest();
            const formData = new FormData();
            formData.append('file', selectedFile);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const upPct = Math.round((e.loaded / e.total) * 100);
                    const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
                    const totalMb = (e.total / (1024 * 1024)).toFixed(1);

                    progressPercent.textContent = `${Math.min(upPct, 99)}%`;
                    progressBarFill.style.width = `${Math.min(upPct * 0.15, 15)}%`;
                    progressStatusText.textContent = `Uploading document to server...`;
                    progressMetaText.textContent = `${loadedMb} / ${totalMb} MB`;
                }
            };

            xhr.onload = () => {
                if (xhr.status === 200) {
                    try {
                        const upRes = JSON.parse(xhr.responseText);
                        if (upRes.success && upRes.session_id) {
                            startPipelineStream(upRes.session_id, upRes.total_pages);
                        } else {
                            throw new Error(upRes.error || 'Upload failed');
                        }
                    } catch (err) {
                        alert(`Upload error: ${err.message}`);
                        resetProgress();
                    }
                } else {
                    alert(`Upload failed with status ${xhr.status}`);
                    resetProgress();
                }
            };

            xhr.onerror = () => {
                alert('Network error during upload.');
                resetProgress();
            };

            xhr.open('POST', '/api/upload');
            xhr.send(formData);
        });
    }


    function startPipelineStream(sessionId, totalPages) {
        progressStageBadge.textContent = 'Processing';
        progressBarFill.style.width = '15%';
        progressStatusText.textContent = 'Analyzing document and starting Indic OCR engine...';
        progressMetaText.textContent = `Total Pages: ${totalPages}`;

        const lang = encodeURIComponent(langSelect.value);
        const dpi = dpiSelect.value;
        const streamUrl = `/api/process-stream/${sessionId}?lang=${lang}&dpi=${dpi}`;

        if (activeEventSource) {
            activeEventSource.close();
        }

        activeEventSource = new EventSource(streamUrl);

        activeEventSource.onmessage = (e) => {
            if (!e.data || e.data.trim() === ': heartbeat') return;

            try {
                const event = JSON.parse(e.data);

                if (event.stage === 'inspecting') {
                    progressStageBadge.textContent = 'Inspecting';
                    progressBarFill.style.width = `${event.percent || 10}%`;
                    progressPercent.textContent = `${event.percent || 10}%`;
                    progressStatusText.textContent = event.message || 'Analyzing document...';
                } else if (event.stage === 'rasterizing') {
                    progressStageBadge.textContent = 'Rasterizing';
                    progressBarFill.style.width = `${event.percent || 15}%`;
                    progressPercent.textContent = `${event.percent || 15}%`;
                    progressStatusText.textContent = event.message || 'Rasterizing pages...';
                } else if (event.stage === 'ocr') {
                    progressStageBadge.textContent = 'Indic OCR';
                    progressBarFill.style.width = `${event.percent}%`;
                    progressPercent.textContent = `${event.percent}%`;
                    progressStatusText.textContent = event.message;
                    progressMetaText.textContent = `Page ${event.current_page} of ${event.total_pages}`;
                } else if (event.stage === 'extracting') {
                    progressStageBadge.textContent = 'Extracting';
                    progressBarFill.style.width = `${event.percent}%`;
                    progressPercent.textContent = `${event.percent}%`;
                    progressStatusText.textContent = event.message;
                } else if (event.stage === 'correcting') {
                    progressStageBadge.textContent = 'Morphology Fix';
                    progressBarFill.style.width = `${event.percent}%`;
                    progressPercent.textContent = `${event.percent}%`;
                    progressStatusText.textContent = event.message;
                } else if (event.stage === 'exporting') {
                    progressStageBadge.textContent = 'Exporting PDF';
                    progressBarFill.style.width = `${event.percent}%`;
                    progressPercent.textContent = `${event.percent}%`;
                    progressStatusText.textContent = event.message;
                } else if (event.stage === 'finalizing') {
                    progressStageBadge.textContent = 'Finalizing';
                    progressBarFill.style.width = '98%';
                    progressPercent.textContent = '98%';
                    progressStatusText.textContent = event.message || 'Preparing report...';
                } else if (event.stage === 'complete' && event.payload) {
                    progressPercent.textContent = '100%';
                    progressBarFill.style.width = '100%';
                    progressStatusText.textContent = 'Done!';
                    activeEventSource.close();
                    setTimeout(() => {
                        resetProgress();
                        renderDocumentResults(event.payload);
                    }, 400);
                } else if (event.stage === 'error') {
                    activeEventSource.close();
                    alert(`Pipeline error: ${event.message || event.error}`);
                    resetProgress();
                }
            } catch (err) {
                console.error("SSE parse error:", err);
            }
        };

        activeEventSource.onerror = (err) => {
            console.error("SSE connection error:", err);
        };
    }

    function resetProgress() {
        processBtn.disabled = false;
        processBtn.textContent = 'Run Ingest & Correction';
        progressCard.style.display = 'none';
        if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
        }
    }

    function renderDocumentResults(data) {
        if (!data) return;
        docResultsCard.style.display = 'block';
        docStatPages.textContent = data.total_pages ?? (data.result ? data.result.total_pages : 0);
        docStatFixes.textContent = data.total_corrections ?? (data.result ? data.result.total_corrections : 0);
        docStatTime.textContent = `${data.latency_seconds || 0}s`;

        docRawDisplay.textContent = data.raw_text || '(No text extracted)';
        docCorrectedDisplay.textContent = data.corrected_text || '(No text extracted)';


        btnDownloadPdf.href = data.download_urls.pdf;
        btnDownloadTxt.href = data.download_urls.txt;
        btnDownloadJson.href = data.download_urls.json;

        // Render detailed corrections
        const corrections = data.result.corrections_summary || [];
        if (corrections.length > 0) {
            docCorrectionsTableBody.innerHTML = corrections.map((c, idx) => `
                <tr>
                    <td style="font-family: var(--font-mono); color: var(--text-tertiary); font-size: 13px;">${idx + 1}</td>
                    <td class="tag-red-strike">${escapeHtml(c.original)}</td>
                    <td class="tag-green-bold">${escapeHtml(c.correction)}</td>
                    <td style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 13px;">${c.edit_distance}</td>
                </tr>
            `).join('');
        } else {
            docCorrectionsTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--apple-emerald); padding: 24px; font-weight: 500;">✓ Document text is completely clean! No corrections were required.</td></tr>';
        }

        docResultsCard.scrollIntoView({ behavior: 'smooth' });
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

        if (tessStatusEl) {
            tessStatusEl.textContent = data.tesseract_available ? 'Active' : 'Standby / Digital Mode';
            tessStatusEl.className = data.tesseract_available ? 'apple-stat-value emerald' : 'apple-stat-value cyan';
        }

        if (dictCountEl) {
            dictCountEl.textContent = data.dictionary_words_count.toLocaleString();
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

function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
