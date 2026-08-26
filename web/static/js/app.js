/**
 * Kannada OCR & Autocorrect — Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initLiveAutocorrect();
    initDocumentUpload();
    initSystemStatus();
});

/* ── Tab Switcher ── */
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

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
            debounceTimer = setTimeout(triggerCorrection, 350);
        });
    }

    async function triggerCorrection() {
        const text = (rawInput.value || '').trim();
        if (!text) {
            correctedDisplay.innerHTML = '<span class="text-dim">Corrected text will appear here in real time...</span>';
            statWords.textContent = '0';
            statFixes.textContent = '0';
            statAccuracy.textContent = '100%';
            statLatency.textContent = '0.00s';
            tableBody.innerHTML = '<tr><td colspan="4" class="text-center text-dim">No corrections performed yet.</td></tr>';
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
                highlighted = highlighted.replace(reg, `<mark class="diff-tag" title="Original: ${escapeHtml(c.original)}">$1</mark>`);
            });
        }
        correctedDisplay.innerHTML = highlighted;

        // Render Table
        if (data.corrections && data.corrections.length > 0) {
            tableBody.innerHTML = data.corrections.map((c, idx) => `
                <tr>
                    <td class="font-mono">${idx + 1}</td>
                    <td class="tag-original">${escapeHtml(c.original)}</td>
                    <td class="tag-corrected">${escapeHtml(c.correction)}</td>
                    <td class="font-mono text-dim">${c.edit_distance}</td>
                </tr>
            `).join('');
        } else {
            tableBody.innerHTML = '<tr><td colspan="4" class="text-center text-emerald">✓ All Kannada words are valid & clean! No errors found.</td></tr>';
        }
    }

    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            if (latestCorrectedText) {
                navigator.clipboard.writeText(latestCorrectedText);
                copyBtn.textContent = '✓ Copied!';
                setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
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

/* ── Tab 2: Document Upload & Full Pipeline ── */
function initDocumentUpload() {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('docFileInput');
    const langSelect = document.getElementById('docLangSelect');
    const dpiSelect = document.getElementById('docDpiSelect');
    const processBtn = document.getElementById('docProcessBtn');
    const progressBar = document.getElementById('docProgressBar');
    const progressStatus = document.getElementById('docProgressStatus');
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
            nameEl.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
            nameEl.style.display = 'block';
        }
        processBtn.disabled = false;
    }

    if (processBtn) {
        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            processBtn.disabled = true;
            progressBar.style.display = 'block';
            progressStatus.textContent = 'Ingesting document and executing Indic OCR...';
            docResultsCard.style.display = 'none';

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('lang', langSelect.value);
            formData.append('dpi', dpiSelect.value);

            try {
                const resp = await fetch('/api/process-document', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();

                if (resp.ok && data.success) {
                    renderDocumentResults(data);
                } else {
                    alert(`Error: ${data.error || 'Failed to process document'}`);
                }
            } catch (err) {
                alert(`Network error: ${err.message}`);
            } finally {
                processBtn.disabled = false;
                progressBar.style.display = 'none';
            }
        });
    }

    function renderDocumentResults(data) {
        docResultsCard.style.display = 'block';
        docStatPages.textContent = data.total_pages;
        docStatFixes.textContent = data.total_corrections;
        docStatTime.textContent = `${data.latency_seconds}s`;

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
                    <td class="font-mono">${idx + 1}</td>
                    <td class="tag-original">${escapeHtml(c.original)}</td>
                    <td class="tag-corrected">${escapeHtml(c.correction)}</td>
                    <td class="font-mono text-dim">${c.edit_distance}</td>
                </tr>
            `).join('');
        } else {
            docCorrectionsTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-emerald">✓ Document text is completely clean! No spelling corrections were required.</td></tr>';
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
                ? `<span class="status-dot"></span> Tesseract OCR Active`
                : `<span style="color:#F43F5E;">●</span> Tesseract Offline (Digital PDFs Only)`;
        }

        const tessStatusEl = document.getElementById('diagTessStatus');
        const dictCountEl = document.getElementById('diagDictCount');
        const langTagsEl = document.getElementById('diagLangTags');

        if (tessStatusEl) {
            tessStatusEl.textContent = data.tesseract_available ? 'Online & Accessible' : 'Not installed in PATH';
            tessStatusEl.className = data.tesseract_available ? 'stat-value success' : 'stat-value amber';
        }

        if (dictCountEl) {
            dictCountEl.textContent = data.dictionary_words_count.toLocaleString();
        }

        if (langTagsEl && data.installed_languages) {
            langTagsEl.innerHTML = data.installed_languages.map(l => `
                <span class="btn btn-secondary btn-sm">${l}</span>
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
