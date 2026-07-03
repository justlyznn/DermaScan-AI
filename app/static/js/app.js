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
    const modelSelect = document.getElementById('model-select');
    
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    
    const resOriginal = document.getElementById('res-original');
    const resPreprocessed = document.getElementById('res-preprocessed');
    const resMask = document.getElementById('res-mask');
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
        
        // Remove active state from samples
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
            // Highlight selected sample
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
        formData.append('model_type', modelSelect.value);
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
            
            const resultsGrid = document.getElementById('results-grid');
            const preprocessedCard = document.getElementById('preprocessed-card');
            
            if (usePrep && usePrep.checked) {
                preprocessedCard.classList.remove('hidden');
                resultsGrid.classList.remove('md:grid-cols-3', 'lg:grid-cols-3');
                resultsGrid.classList.add('md:grid-cols-2', 'lg:grid-cols-4');
            } else {
                preprocessedCard.classList.add('hidden');
                resultsGrid.classList.remove('md:grid-cols-2', 'lg:grid-cols-4');
                resultsGrid.classList.add('md:grid-cols-3', 'lg:grid-cols-3');
            }
            
            // Render images
            resOriginal.src = 'data:image/jpeg;base64,' + data.original;
            resPreprocessed.src = 'data:image/jpeg;base64,' + data.preprocessed;
            resMask.src = 'data:image/jpeg;base64,' + data.mask;
            resGradcam.src = 'data:image/jpeg;base64,' + data.gradcam;

            // Show results
            loadingState.classList.add('hidden');
            loadingState.classList.remove('flex');
            resultsSection.classList.remove('hidden');
            actionBar.classList.remove('hidden'); // allow re-run with different model

        } catch (error) {
            alert('Error: ' + error.message);
            loadingState.classList.add('hidden');
            loadingState.classList.remove('flex');
            actionBar.classList.remove('hidden');
        }
    });
});
