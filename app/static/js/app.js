document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewImage = document.getElementById('preview-image');
    const removeBtnContainer = document.getElementById('remove-btn-container');
    const removeBtn = document.getElementById('remove-btn');
    const sampleImgs = document.querySelectorAll('.sample-img');
    
    const actionBar = document.getElementById('action-bar');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    
    const resOriginal = document.getElementById('res-original');
    const resGradcam = document.getElementById('res-gradcam');

    let currentFile = null;
    let currentSamplePath = null;

    // --- Drag and Drop Handlers ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-emerald-500', 'bg-emerald-100');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-emerald-500', 'bg-emerald-100');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-emerald-500', 'bg-emerald-100');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Harap unggah file gambar (JPG/PNG).');
            return;
        }
        currentFile = file;
        currentSamplePath = null;
        
        sampleImgs.forEach(img => img.classList.remove('border-emerald-500'));

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            showPreview();
        };
        reader.readAsDataURL(file);
    }

    // --- Sample Selection ---
    sampleImgs.forEach(img => {
        img.addEventListener('click', () => {
            sampleImgs.forEach(s => s.classList.remove('border-emerald-500'));
            img.classList.add('border-emerald-500');
            
            currentFile = null;
            currentSamplePath = img.getAttribute('data-path');
            
            previewImage.src = currentSamplePath;
            showPreview();
        });
    });

    function showPreview() {
        uploadContent.classList.add('hidden');
        previewImage.classList.remove('hidden');
        removeBtnContainer.classList.remove('hidden');
        actionBar.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    }

    // --- Remove Image ---
    removeBtn.addEventListener('click', () => {
        currentFile = null;
        currentSamplePath = null;
        fileInput.value = '';
        
        previewImage.src = '';
        previewImage.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        removeBtnContainer.classList.add('hidden');
        actionBar.classList.add('hidden');
        resultsSection.classList.add('hidden');
        
        sampleImgs.forEach(s => s.classList.remove('border-emerald-500'));
    });

    // --- Analyze Button ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile && !currentSamplePath) return;

        // UI State
        actionBar.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');
        loadingState.classList.add('flex');

        const usePrep = document.getElementById('use-preprocessing');

        const formData = new FormData();
        // Always send model_type as 'cbam' (single model)
        formData.append('model_type', 'cbam');
        if (usePrep) {
            formData.append('use_preprocessing', usePrep.checked);
        }

        if (currentFile) {
            formData.append('file', currentFile);
        } else if (currentSamplePath) {
            formData.append('sample_path', currentSamplePath);
        }

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Terjadi kesalahan pada server');
            }

            const data = await response.json();
            
            // Set images on slider
            resOriginal.src = 'data:image/jpeg;base64,' + data.original;
            resGradcam.src  = 'data:image/jpeg;base64,' + data.gradcam;

            // Update preprocessing info panel
            const infoPrepStatus = document.getElementById('info-prep-status');
            if (infoPrepStatus) {
                if (usePrep && usePrep.checked) {
                    infoPrepStatus.textContent = 'DullRazor + CLAHE Aktif';
                } else {
                    infoPrepStatus.textContent = 'Preprocessing Dinonaktifkan';
                }
            }

            // Wait for images to load, then init slider
            await Promise.all([
                waitForImage(resOriginal),
                waitForImage(resGradcam)
            ]);

            initComparisonSlider('slider-gradcam');

            // Show results
            loadingState.classList.add('hidden');
            loadingState.classList.remove('flex');
            resultsSection.classList.remove('hidden');
            actionBar.classList.remove('hidden');

        } catch (error) {
            alert('Error: ' + error.message);
            loadingState.classList.add('hidden');
            loadingState.classList.remove('flex');
            actionBar.classList.remove('hidden');
        }
    });

    // --- Helper: wait for image to load ---
    function waitForImage(imgEl) {
        return new Promise((resolve) => {
            if (imgEl.complete && imgEl.naturalWidth > 0) {
                resolve();
            } else {
                imgEl.onload = () => resolve();
                imgEl.onerror = () => resolve(); // resolve anyway on error
            }
        });
    }

    // --- Before/After Comparison Slider logic ---
    function initComparisonSlider(wrapperId) {
        const wrapper = document.getElementById(wrapperId);
        if (!wrapper) return;

        const afterEl  = wrapper.querySelector('.comparison-after');
        const handleEl = wrapper.querySelector('.comparison-handle');
        const rangeEl  = wrapper.querySelector('.comparison-range');

        // Reset to 50%
        rangeEl.value = 50;
        setSliderPosition(50, afterEl, handleEl);

        // Remove previous listener if any (re-init on repeated runs)
        const newRange = rangeEl.cloneNode(true);
        rangeEl.parentNode.replaceChild(newRange, rangeEl);

        newRange.addEventListener('input', (e) => {
            const pct = parseFloat(e.target.value);
            setSliderPosition(pct, afterEl, handleEl);
        });
    }

    function setSliderPosition(pct, afterEl, handleEl) {
        // Clip the "after" image: reveal from right
        afterEl.style.clipPath = `inset(0 0 0 ${pct}%)`;
        // Move the handle
        handleEl.style.left = `${pct}%`;
    }
});
