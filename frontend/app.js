// Application State
const state = {
    currentStep: 1,
    totalSteps: 7,
    selectedTask: null,
    selectedMode: null,
    selectedExecution: 'local',
    uploadedFile: null,
    selectedDataset: null,
    targetColumn: null,
    problemType: 'auto',
    validationStrategy: 'train_test_split',
    selectedFeatureEngineering: {
        encoding: [],
        scaling: [],
        imputation: [],
        selection: [],
        interactions: [],
        dimensionality: []
    },
    selectedModel: null,
    sessionId: null,
    websocket: null,
    analysisResults: null,
    trainingStartTime: null,
    isConnected: false
};

// API Configuration
const API = {
    baseURL: 'http://localhost:8000',
    endpoints: {
        upload: '/api/v1/upload-dataset',
        configure: '/api/v1/configure-analysis',
        startTraining: '/api/v1/start-training',
        status: '/api/v1/training-status',
        results: '/api/v1/results',
        websocket: '/ws/training'
    }
};

// Sample data for fallback/demo mode
const sampleData = {
    datasets: [
        { 
            name: "iris", 
            rows: 150, 
            columns: 5, 
            task: "classification", 
            target: "species",
            quality: 0.95,
            missingPercent: 0
        },
        { 
            name: "housing", 
            rows: 506, 
            columns: 14, 
            task: "regression", 
            target: "price",
            quality: 0.87,
            missingPercent: 2.3
        },
        { 
            name: "sales", 
            rows: 365, 
            columns: 3, 
            task: "time_series", 
            target: "sales",
            quality: 0.92,
            missingPercent: 0.8
        }
    ],
    featureRecommendations: {
        categorical: ['onehot', 'target'],
        numerical: ['standard', 'robust'],
        missing: ['simple', 'knn'],
        selection: ['statistical']
    },
    modelRecommendations: [
        {
            name: "Random Forest",
            confidence: 0.94,
            description: "Excellent for this dataset size and type. Handles mixed data types well.",
            pros: ["High accuracy", "Handles missing values", "Feature importance", "Robust to outliers"],
            estimatedTime: "3-5 minutes"
        },
        {
            name: "Gradient Boosting",
            confidence: 0.91,
            description: "Superior performance on structured data with advanced optimization.",
            pros: ["High performance", "Robust to outliers", "Good with small datasets"],
            estimatedTime: "5-8 minutes"
        },
        {
            name: "XGBoost",
            confidence: 0.89,
            description: "State-of-the-art gradient boosting with excellent performance.",
            pros: ["Top performance", "GPU support", "Advanced features"],
            estimatedTime: "4-7 minutes"
        }
    ]
};

// Chart instances
let featureChart = null;
let performanceChart = null;
let trainingChart = null;
let distributionChart = null;

// DOM Elements
const elements = {
    // Navigation
    prevBtn: document.getElementById('prevBtn'),
    nextBtn: document.getElementById('nextBtn'),
    
    // Connection status
    connectionStatus: document.getElementById('connectionStatus'),
    
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressFill: document.getElementById('progressFill'),
    uploadStatus: document.getElementById('uploadStatus'),
    
    // Preview
    dataPreview: document.getElementById('dataPreview'),
    rowCount: document.getElementById('rowCount'),
    colCount: document.getElementById('colCount'),
    missingPercent: document.getElementById('missingPercent'),
    dataQuality: document.getElementById('dataQuality'),
    previewTable: document.getElementById('previewTable'),
    dataInsights: document.getElementById('dataInsights'),
    
    // Configuration
    targetColumn: document.getElementById('targetColumn'),
    problemType: document.getElementById('problemType'),
    validationStrategy: document.getElementById('validationStrategy'),
    
    // Feature Engineering
    autoRecommendations: document.getElementById('autoRecommendations'),
    recommendedMethods: document.getElementById('recommendedMethods'),
    applyRecommendations: document.getElementById('applyRecommendations'),
    selectedMethods: document.getElementById('selectedMethods'),
    methodsSummary: document.getElementById('methodsSummary'),
    
    // Training
    manualSection: document.getElementById('manualSection'),
    trainingSection: document.getElementById('trainingSection'),
    recommendationsGrid: document.getElementById('recommendationsGrid'),
    progressValue: document.getElementById('progressValue'),
    currentStage: document.getElementById('currentStage'),
    stageDescription: document.getElementById('stageDescription'),
    estimatedTime: document.getElementById('estimatedTime'),
    sessionId: document.getElementById('sessionId'),
    executionEnv: document.getElementById('executionEnv'),
    stageFill: document.getElementById('stageFill'),
    stageProgress: document.getElementById('stageProgress'),
    logsContent: document.getElementById('logsContent'),
    logEntries: document.getElementById('logEntries'),
    toggleLogs: document.getElementById('toggleLogs'),
    pauseTraining: document.getElementById('pauseTraining'),
    cancelTraining: document.getElementById('cancelTraining'),
    
    // Results
    trainingTime: document.getElementById('trainingTime'),
    bestModel: document.getElementById('bestModel'),
    finalDataQuality: document.getElementById('finalDataQuality'),
    metricsGrid: document.getElementById('metricsGrid'),
    keyFindings: document.getElementById('keyFindings'),
    dataInsightsList: document.getElementById('dataInsightsList'),
    recommendationsList: document.getElementById('recommendationsList'),
    appliedMethods: document.getElementById('appliedMethods'),
    hyperParameters: document.getElementById('hyperParameters'),
    downloadResults: document.getElementById('downloadResults'),
    deployModel: document.getElementById('deployModel'),
    downloadModel: document.getElementById('downloadModel'),
    startNew: document.getElementById('startNew'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer'),
    loadingOverlay: document.getElementById('loadingOverlay')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkBackendConnection();
    updateNavigationButtons();
    updateProgressBar();
    populateTaskDisplayNames();
});

// API Functions
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API.baseURL}/health`, { 
            method: 'GET',
            timeout: 5000 
        });
        
        if (response.ok) {
            state.isConnected = true;
            updateConnectionStatus('🟢', 'Connected to Backend');
            showToast('Backend connection established', 'success');
        } else {
            throw new Error('Backend not responding');
        }
    } catch (error) {
        state.isConnected = false;
        updateConnectionStatus('🟡', 'Demo Mode');
        showToast('Using demo mode - backend not available', 'warning');
        console.warn('Backend connection failed:', error);
    }
}

function updateConnectionStatus(indicator, text) {
    if (elements.connectionStatus) {
        elements.connectionStatus.textContent = indicator;
        elements.connectionStatus.parentElement.querySelector('.status-text').textContent = text;
        
        if (indicator === '🟢') {
            elements.connectionStatus.parentElement.classList.add('connected');
        }
    }
}

async function uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoadingOverlay('Uploading dataset...');
        
        const response = await fetch(`${API.baseURL}${API.endpoints.upload}`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        hideLoadingOverlay();
        
        return result;
    } catch (error) {
        hideLoadingOverlay();
        console.error('Upload error:', error);
        
        // Fallback to mock data for demo
        return {
            session_id: generateSessionId(),
            dataset_info: {
                rows: Math.floor(Math.random() * 1000) + 100,
                columns: Math.floor(Math.random() * 20) + 5,
                missing_percent: (Math.random() * 10).toFixed(1),
                quality_score: (0.7 + Math.random() * 0.3).toFixed(2),
                target_suggestions: ['target', 'label', 'class', 'price', 'value']
            }
        };
    }
}

async function configureAnalysis(config) {
    try {
        showLoadingOverlay('Configuring analysis...');
        
        const response = await fetch(`${API.baseURL}${API.endpoints.configure}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            throw new Error(`Configuration failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        hideLoadingOverlay();
        
        return result;
    } catch (error) {
        hideLoadingOverlay();
        console.error('Configuration error:', error);
        
        // Fallback for demo
        return {
            success: true,
            recommendations: sampleData.featureRecommendations,
            model_suggestions: sampleData.modelRecommendations
        };
    }
}

async function startTraining(config) {
    try {
        showLoadingOverlay('Starting training...');
        
        const response = await fetch(`${API.baseURL}${API.endpoints.startTraining}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            throw new Error(`Training start failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        hideLoadingOverlay();
        
        return result;
    } catch (error) {
        hideLoadingOverlay();
        console.error('Training start error:', error);
        
        // Fallback for demo
        return {
            session_id: state.sessionId || generateSessionId(),
            status: 'started',
            estimated_time: '5-10 minutes'
        };
    }
}

// WebSocket Functions
function connectWebSocket(sessionId) {
    if (state.websocket) {
        state.websocket.close();
    }
    
    try {
        const wsURL = `${API.baseURL.replace('http', 'ws')}${API.endpoints.websocket}/${sessionId}`;
        state.websocket = new WebSocket(wsURL);
        
        state.websocket.onopen = () => {
            console.log('WebSocket connected');
            showToast('Real-time monitoring connected', 'success');
        };
        
        state.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleTrainingUpdate(data);
        };
        
        state.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            // Attempt to reconnect if training is still in progress
            if (state.currentStep === 6) {
                setTimeout(() => connectWebSocket(sessionId), 3000);
            }
        };
        
        state.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            showToast('Real-time monitoring unavailable', 'warning');
            // Fall back to polling
            startPollingUpdates(sessionId);
        };
        
    } catch (error) {
        console.error('WebSocket connection failed:', error);
        startPollingUpdates(sessionId);
    }
}

function startPollingUpdates(sessionId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API.baseURL}${API.endpoints.status}/${sessionId}`);
            if (response.ok) {
                const data = await response.json();
                handleTrainingUpdate(data);
                
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(pollInterval);
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

function handleTrainingUpdate(data) {
    updateTrainingProgress(data.progress || 0);
    updateCurrentStage(data.current_stage || 'processing', data.stage_description || '');
    updateEstimatedTime(data.estimated_time_remaining || 'calculating...');
    
    if (data.logs) {
        appendLogs(data.logs);
    }
    
    if (data.status === 'completed') {
        handleTrainingComplete(data);
    } else if (data.status === 'failed') {
        handleTrainingFailure(data.error || 'Training failed');
    }
}

function handleTrainingComplete(data) {
    state.analysisResults = data;
    showToast('Training completed successfully!', 'success');
    
    // Automatically proceed to results
    setTimeout(() => {
        goToNextStep();
    }, 1000);
}

function handleTrainingFailure(error) {
    showToast(`Training failed: ${error}`, 'error');
    elements.currentStage.textContent = 'Training Failed';
    elements.stageDescription.textContent = error;
}

// Event Listeners
function initializeEventListeners() {
    // Navigation
    elements.prevBtn.addEventListener('click', goToPreviousStep);
    elements.nextBtn.addEventListener('click', goToNextStep);

    // Task Selection
    document.querySelectorAll('.task-card').forEach(card => {
        card.addEventListener('click', () => selectTask(card.dataset.task));
    });

    // Mode Selection
    document.querySelectorAll('.mode-card').forEach(card => {
        card.addEventListener('click', () => selectMode(card.dataset.mode));
    });

    // Execution Mode Selection
    document.querySelectorAll('.execution-option').forEach(option => {
        option.addEventListener('click', () => selectExecution(option.dataset.execution));
    });

    // File Upload
    elements.browseBtn?.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput?.addEventListener('change', handleFileUpload);
    elements.uploadArea?.addEventListener('dragover', handleDragOver);
    elements.uploadArea?.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea?.addEventListener('drop', handleFileDrop);

    // Sample Datasets
    document.querySelectorAll('.dataset-card').forEach(card => {
        card.addEventListener('click', () => selectSampleDataset(card.dataset.dataset));
    });

    // Configuration
    elements.targetColumn?.addEventListener('change', (e) => {
        state.targetColumn = e.target.value;
        updateNavigationButtons();
    });

    elements.problemType?.addEventListener('change', (e) => {
        state.problemType = e.target.value;
    });

    elements.validationStrategy?.addEventListener('change', (e) => {
        state.validationStrategy = e.target.value;
    });

    // Feature Engineering
    elements.applyRecommendations?.addEventListener('click', applyAllRecommendations);

    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', handleFeatureEngineeringSelection);
    });

    // Training Controls
    elements.toggleLogs?.addEventListener('click', toggleTrainingLogs);
    elements.pauseTraining?.addEventListener('click', pauseTraining);
    elements.cancelTraining?.addEventListener('click', cancelTraining);

    // Results Actions
    elements.downloadResults?.addEventListener('click', downloadResults);
    elements.deployModel?.addEventListener('click', deployModel);
    elements.downloadModel?.addEventListener('click', downloadModel);
    elements.startNew?.addEventListener('click', startNewAnalysis);
}

// Navigation Functions
function goToNextStep() {
    if (canProceedToNextStep()) {
        if (state.currentStep < state.totalSteps) {
            state.currentStep++;
            updateStepDisplay();
            updateNavigationButtons();
            updateProgressBar();
            handleStepTransition();
        }
    }
}

function goToPreviousStep() {
    if (state.currentStep > 1) {
        state.currentStep--;
        updateStepDisplay();
        updateNavigationButtons();
        updateProgressBar();
    }
}

function canProceedToNextStep() {
    switch (state.currentStep) {
        case 1:
            if (!state.selectedTask) {
                showToast('Please Select A Task', 'warning');
                return false;
            }
            return true;
        case 2:
            if (!state.selectedMode) {
                showToast('Please Select A Mode', 'warning');
                return false;
            }
            return true;
        case 3:
            if (!state.uploadedFile && !state.selectedDataset) {
                showToast('Please Upload A File Or Select A Sample Dataset', 'warning');
                return false;
            }
            return true;
        case 4:
            if (requiresTargetColumn() && !state.targetColumn) {
                showToast('कृपया टारगेट कॉलम चुनें', 'warning');
                return false;
            }
            return true;
        case 5:
            const hasSelectedMethods = Object.values(state.selectedFeatureEngineering).some(arr => arr.length > 0);
            if (state.selectedTask !== 'automated_ml' && !hasSelectedMethods) {
                showToast('Select At Least One Feature Engineering Method', 'warning');
                return false;
            }
            return true;
        case 6:
            if (state.selectedMode === 'training' && !state.selectedModel && state.selectedTask === 'manual_ml') {
                showToast('Please Select A Model', 'warning');
                return false;
            }
            return true;
        default:
            return true;
    }
}

function updateStepDisplay() {
    document.querySelectorAll('.step-container').forEach(container => {
        container.classList.remove('active');
    });
    
    const currentStepContainer = document.getElementById(`step${state.currentStep}`);
    if (currentStepContainer) {
        currentStepContainer.classList.add('active');
    }
}

function updateNavigationButtons() {
    elements.prevBtn.disabled = state.currentStep === 1;
    
    if (state.currentStep === state.totalSteps) {
        elements.nextBtn.style.display = 'none';
    } else {
        elements.nextBtn.style.display = 'inline-flex';
        elements.nextBtn.disabled = !canProceedToNextStep();
    }
}

function updateProgressBar() {
    document.querySelectorAll('.progress-step').forEach((step, index) => {
        const stepNumber = index + 1;
        step.classList.remove('active', 'completed');
        
        if (stepNumber === state.currentStep) {
            step.classList.add('active');
        } else if (stepNumber < state.currentStep) {
            step.classList.add('completed');
        }
    });
}

async function handleStepTransition() {
    switch (state.currentStep) {
        case 4:
            updateConfigurationSummary();
            break;
        case 5:
            await loadFeatureEngineering();
            break;
        case 6:
            if (state.selectedTask === 'automated_ml' || state.selectedMode === 'visualization') {
                await startAnalysis();
            } else {
                await showModelRecommendations();
            }
            break;
        case 7:
            displayResults();
            break;
    }
}

// Task Selection
function selectTask(taskType) {
    state.selectedTask = taskType;
    
    document.querySelectorAll('.task-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    document.querySelector(`[data-task="${taskType}"]`).classList.add('selected');
    updateNavigationButtons();
    
    showToast(`Task Selected: ${getTaskDisplayName(taskType)}`, 'success');
}

function selectMode(mode) {
    state.selectedMode = mode;
    
    document.querySelectorAll('.mode-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    const selectedCard = document.querySelector(`[data-mode="${mode}"]`);
    selectedCard.classList.add('selected');
    
    // Show execution options for training mode
    if (mode === 'training') {
        selectedCard.querySelector('.execution-options').style.display = 'flex';
        // Auto-select local by default
        selectExecution('local');
    }
    
    updateNavigationButtons();
    
    const modeDisplayName = mode === 'visualization' ? 'Only Visualization' : 'Model Training';
    showToast(`Mode Selected: ${modeDisplayName}`, 'success');
}

function selectExecution(execution) {
    state.selectedExecution = execution;
    
    document.querySelectorAll('.execution-option').forEach(option => {
        option.classList.remove('selected');
    });
    
    document.querySelector(`[data-execution="${execution}"]`).classList.add('selected');
}

function getTaskDisplayName(taskType) {
    const taskNames = {
        'eda': 'Exploratory Data Analysis',
        'manual_ml': 'Manual Machine Learning',
        'automated_ml': 'Automated Machine Learning',
        'clustering': 'Clustering',
        'anomaly': 'Anomaly Detection',
        'timeseries': 'Time Series Analysis'
    };
    return taskNames[taskType] || taskType;
}

// File Upload Handlers
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        await processUploadedFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

async function handleFileDrop(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file) {
        await processUploadedFile(file);
    }
}

async function processUploadedFile(file) {
    // Validate file
    const allowedTypes = ['.csv', '.xlsx', '.xls', '.json'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        showToast('Please Upload CSV, Excel, Or JSON File', 'error');
        return;
    }
    
    if (file.size > 100 * 1024 * 1024) {
        showToast('File Size Must Be Less Than 100MB', 'error');
        return;
    }
    
    // Show upload progress
    elements.uploadProgress.classList.remove('hidden');
    elements.uploadStatus.textContent = 'Uploading...';
    
    // Simulate progress for demo
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 10;
        elements.progressFill.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            elements.uploadStatus.textContent = 'File Uploaded!';
        }
    }, 200);
    
    try {
        // Upload file to backend
        const result = await uploadDataset(file);
        
        state.uploadedFile = file;
        state.selectedDataset = null;
        state.sessionId = result.session_id;
        
        // Clear dataset selection
        document.querySelectorAll('.dataset-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        await showDataPreview(file, result.dataset_info);
        showToast(`File Uploaded: ${file.name}`, 'success');
        updateNavigationButtons();
        
    } catch (error) {
        showToast('Issue In File Upload', 'error');
        console.error('Upload failed:', error);
    } finally {
        setTimeout(() => {
            elements.uploadProgress.classList.add('hidden');
            elements.progressFill.style.width = '0%';
        }, 1000);
    }
}

async function selectSampleDataset(datasetName) {
    state.selectedDataset = datasetName;
    state.uploadedFile = null;
    state.sessionId = generateSessionId();
    
    document.querySelectorAll('.dataset-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    document.querySelector(`[data-dataset="${datasetName}"]`).classList.add('selected');
    
    const dataset = sampleData.datasets.find(d => d.name === datasetName);
    await showDataPreview(null, dataset);
    showToast(`Database Chosen: ${datasetName}`, 'success');
    updateNavigationButtons();
}

async function showDataPreview(file, datasetInfo) {
    const preview = elements.dataPreview;
    
    let rows, columns, missing, quality, datasetName;
    
    if (file && datasetInfo) {
        rows = datasetInfo.rows;
        columns = datasetInfo.columns;
        missing = datasetInfo.missing_percent + '%';
        quality = (datasetInfo.quality_score * 100).toFixed(0) + '%';
        datasetName = 'uploaded';
    } else if (datasetInfo) {
        rows = datasetInfo.rows;
        columns = datasetInfo.columns;
        missing = datasetInfo.missingPercent + '%';
        quality = (datasetInfo.quality * 100).toFixed(0) + '%';
        datasetName = datasetInfo.name;
    }
    
    elements.rowCount.textContent = rows.toLocaleString();
    elements.colCount.textContent = columns;
    elements.missingPercent.textContent = missing;
    elements.dataQuality.textContent = quality;
    
    // Generate sample table
    generateSampleTable(elements.previewTable, datasetName);
    
    // Show data insights
    generateDataInsights(elements.dataInsights, datasetName);
    
    preview.classList.remove('hidden');
    
    // Populate target column dropdown
    setTimeout(() => {
        populateTargetColumnDropdown(datasetName, datasetInfo);
    }, 100);
}

function generateSampleTable(container, datasetName) {
    let headers, sampleRows;
    
    if (datasetName === 'iris') {
        headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'];
        sampleRows = [
            [5.1, 3.5, 1.4, 0.2, 'setosa'],
            [4.9, 3.0, 1.4, 0.2, 'setosa'],
            [7.0, 3.2, 4.7, 1.4, 'versicolor'],
            [6.4, 3.2, 4.5, 1.5, 'versicolor'],
            [6.3, 3.3, 6.0, 2.5, 'virginica']
        ];
    } else if (datasetName === 'housing') {
        headers = ['area', 'bedrooms', 'bathrooms', 'stories', 'price'];
        sampleRows = [
            [7420, 4, 2, 3, 13300000],
            [8960, 4, 4, 4, 12250000],
            [9960, 3, 2, 2, 12250000],
            [7500, 4, 2, 2, 11410000],
            [7420, 4, 1, 2, 10850000]
        ];
    } else if (datasetName === 'sales') {
        headers = ['date', 'product', 'sales'];
        sampleRows = [
            ['2023-01-01', 'Product A', 1250],
            ['2023-01-02', 'Product A', 1180],
            ['2023-01-03', 'Product A', 1320],
            ['2023-01-04', 'Product A', 1450],
            ['2023-01-05', 'Product A', 1380]
        ];
    } else {
        headers = ['column_1', 'column_2', 'column_3', 'column_4', 'target'];
        sampleRows = [
            [12.5, 'Category A', 0.85, 45, 1],
            [8.3, 'Category B', 0.72, 38, 0],
            [15.7, 'Category A', 0.91, 52, 1],
            [6.2, 'Category C', 0.64, 29, 0],
            [11.8, 'Category B', 0.88, 41, 1]
        ];
    }
    
    const table = document.createElement('table');
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    sampleRows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    container.innerHTML = '';
    container.appendChild(table);
}

function generateDataInsights(container, datasetName) {
    const insights = [
        { icon: '📊', text: 'Target Column Suggested' },
        { icon: '🎯', text: 'Target Column Suggested' },
        { icon: '🔢', text: 'Both Numerical And Categorical Features Found' },
        { icon: '✅', text: 'Useful for Machine Learning' }
    ];
    
    container.innerHTML = '<h4>Data Insights</h4>';
    
    insights.forEach(insight => {
        const item = document.createElement('div');
        item.className = 'insight-item';
        item.innerHTML = `
            <span class="insight-icon">${insight.icon}</span>
            <span>${insight.text}</span>
        `;
        container.appendChild(item);
    });
}

function populateTargetColumnDropdown(datasetName, datasetInfo) {
    const select = elements.targetColumn;
    if (!select) return;
    
    // Clear existing options
    select.innerHTML = '<option value="">ऑटो-डिटेक्ट टारगेट कॉलम...</option>';
    
    let columns = [];
    if (datasetName === 'iris') {
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'];
    } else if (datasetName === 'housing') {
        columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'price'];
    } else if (datasetName === 'sales') {
        columns = ['date', 'product', 'sales'];
    } else if (datasetInfo && datasetInfo.target_suggestions) {
        columns = datasetInfo.target_suggestions;
    } else {
        columns = ['column_1', 'column_2', 'column_3', 'column_4', 'target'];
    }
    
    // Add all columns as options
    columns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        select.appendChild(option);
    });
    
    // Auto-select the typical target column for known datasets
    if (datasetName === 'iris') {
        select.value = 'species';
        state.targetColumn = 'species';
    } else if (datasetName === 'housing') {
        select.value = 'price';
        state.targetColumn = 'price';
    } else if (datasetName === 'sales') {
        select.value = 'sales';
        state.targetColumn = 'sales';
    }
    
    updateNavigationButtons();
}

function requiresTargetColumn() {
    return ['manual_ml', 'automated_ml', 'timeseries'].includes(state.selectedTask);
}

// Configuration
function updateConfigurationSummary() {
    document.getElementById('selectedTask').textContent = getTaskDisplayName(state.selectedTask);
    document.getElementById('selectedMode').textContent = state.selectedMode === 'visualization' ? 'Visualization' : 'Model Training';
    document.getElementById('selectedExecution').textContent = state.selectedExecution === 'local' ? 'Local' : 'Kaggle';
    document.getElementById('selectedDataset').textContent = state.uploadedFile ? state.uploadedFile.name : state.selectedDataset;
    
    // Determine problem type
    let problemType = 'Unknown';
    if (state.selectedDataset) {
        const dataset = sampleData.datasets.find(d => d.name === state.selectedDataset);
        if (dataset) {
            const typeMap = {
                'classification': 'Classification',
                'regression': 'Regression',
                'time_series': 'Time Series'
            };
            problemType = typeMap[dataset.task] || dataset.task;
        }
    } else if (state.selectedTask === 'clustering') {
        problemType = 'Clustering';
    } else if (state.selectedTask === 'anomaly') {
        problemType = 'Anomaly Detection';
    } else if (state.selectedTask === 'timeseries') {
        problemType = 'Time Series';
    }
    
    document.getElementById('detectedProblemType').textContent = problemType;
}

// Feature Engineering
async function loadFeatureEngineering() {
    if (state.selectedTask === 'automated_ml') {
        // Skip feature engineering selection for automated ML
        document.getElementById('featureEngineeringDescription').textContent = 
            'All Feature Engineering Tasks Will Be Automatically Done In Automated ML';
        return;
    }
    
    // Load recommendations
    try {
        const config = {
            session_id: state.sessionId,
            task_type: state.selectedTask,
            target_column: state.targetColumn,
            problem_type: state.problemType
        };
        
        const result = await configureAnalysis(config);
        showFeatureRecommendations(result.recommendations || sampleData.featureRecommendations);
        
    } catch (error) {
        console.error('Failed to load feature engineering recommendations:', error);
        showFeatureRecommendations(sampleData.featureRecommendations);
    }
}

function showFeatureRecommendations(recommendations) {
    const container = elements.recommendedMethods;
    container.innerHTML = '';
    
    // Show recommended methods
    Object.keys(recommendations).forEach(category => {
        recommendations[category].forEach(method => {
            const tag = document.createElement('span');
            tag.className = 'recommended-method';
            tag.textContent = method;
            container.appendChild(tag);
        });
    });
}

function applyAllRecommendations() {
    // Auto-select recommended methods
    const recommendations = sampleData.featureRecommendations;
    
    Object.keys(recommendations).forEach(category => {
        const categoryMap = {
            'categorical': 'encoding',
            'numerical': 'scaling',
            'missing': 'imputation',
            'selection': 'selection'
        };
        
        const targetCategory = categoryMap[category] || category;
        
        recommendations[category].forEach(method => {
            const checkbox = document.querySelector(`input[name="${targetCategory}"][value="${method}"]`);
            if (checkbox) {
                checkbox.checked = true;
                // Trigger change event
                checkbox.dispatchEvent(new Event('change'));
            }
        });
    });
    
    showToast('All Suggested Methods Applied', 'success');
}

function handleFeatureEngineeringSelection(event) {
    const { name, value, checked } = event.target;
    
    if (checked) {
        if (!state.selectedFeatureEngineering[name].includes(value)) {
            state.selectedFeatureEngineering[name].push(value);
        }
    } else {
        state.selectedFeatureEngineering[name] = state.selectedFeatureEngineering[name].filter(v => v !== value);
    }
    
    updateSelectedMethodsSummary();
    updateNavigationButtons();
}

function updateSelectedMethodsSummary() {
    const container = elements.methodsSummary;
    container.innerHTML = '';
    
    const allMethods = [];
    Object.keys(state.selectedFeatureEngineering).forEach(category => {
        state.selectedFeatureEngineering[category].forEach(method => {
            allMethods.push({ category, method });
        });
    });
    
    if (allMethods.length === 0) {
        container.innerHTML = '<p class="no-methods">No Methods Selected</p>';
        return;
    }
    
    allMethods.forEach(item => {
        const tag = document.createElement('div');
        tag.className = 'selected-method-tag';
        tag.innerHTML = `
            <span>${item.method}</span>
            <button class="remove-method" data-category="${item.category}" data-method="${item.method}">×</button>`;
        
        tag.querySelector('.remove-method').addEventListener('click', (e) => {
            const { category, method } = e.target.dataset;
            removeSelectedMethod(category, method);
        });
        
        container.appendChild(tag);
    });
}

function removeSelectedMethod(category, method) {
    const checkbox = document.querySelector(`input[name="${category}"][value="${method}"]`);
    if (checkbox) {
        checkbox.checked = false;
        checkbox.dispatchEvent(new Event('change'));
    }
}

// Model Selection and Training
async function showModelRecommendations() {
    elements.manualSection.classList.remove('hidden');
    elements.trainingSection.classList.add('hidden');
    
    const grid = elements.recommendationsGrid;
    grid.innerHTML = '';
    
    // Get model recommendations from backend or use sample data
    const recommendations = sampleData.modelRecommendations;
    
    recommendations.forEach(model => {
        const card = createModelRecommendationCard(model);
        grid.appendChild(card);
    });
}

function createModelRecommendationCard(model) {
    const card = document.createElement('div');
    card.className = 'recommendation-card';
    card.dataset.model = model.name;
    
    card.innerHTML = `
        <div class="recommendation-header">
            <h4 class="model-name">${model.name}</h4>
            <span class="confidence-score">${Math.round(model.confidence * 100)}%</span>
        </div>
        <p class="model-description">${model.description}</p>
        <div class="model-pros">
            ${model.pros.map(pro => `<span class="pro-tag">${pro}</span>`).join('')}
        </div>
        <div style="margin-top: 12px; font-size: 12px; color: var(--color-text-secondary);">
            Estimated Time: ${model.estimatedTime}
        </div>
    `;
    
    card.addEventListener('click', () => selectModel(model.name, card));
    
    return card;
}

function selectModel(modelName, cardElement) {
    state.selectedModel = modelName;
    
    document.querySelectorAll('.recommendation-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    cardElement.classList.add('selected');
    updateNavigationButtons();
    
    showToast(`मॉडल चुना गया: ${modelName}`, 'success');
}

async function startAnalysis() {
    elements.manualSection.classList.add('hidden');
    elements.trainingSection.classList.remove('hidden');
    
    // Set session ID in display
    elements.sessionId.textContent = state.sessionId || generateSessionId();
    
    // Update execution environment
    const envIcon = state.selectedExecution === 'kaggle' ? '🏆' : '💻';
    const envText = state.selectedExecution === 'kaggle' ? 'Kaggle Environment' : 'Local Machine';
    elements.executionEnv.innerHTML = `
        <span class="env-icon">${envIcon}</span>
        <span class="env-text">${envText}</span>
    `;
    
    // Prepare training config
    const config = {
        session_id: state.sessionId,
        task_type: state.selectedTask,
        mode: state.selectedMode,
        execution: state.selectedExecution,
        target_column: state.targetColumn,
        problem_type: state.problemType,
        validation_strategy: state.validationStrategy,
        feature_engineering: state.selectedFeatureEngineering,
        selected_model: state.selectedModel,
        automated: state.selectedTask === 'automated_ml'
    };
    
    try {
        // Start training
        const result = await startTraining(config);
        state.sessionId = result.session_id;
        state.trainingStartTime = Date.now();
        
        // Connect WebSocket for real-time updates
        connectWebSocket(state.sessionId);
        
        // Start simulation if backend not available
        if (!state.isConnected) {
            simulateTraining();
        }
        
        showToast('Training Started!', 'success');
        
    } catch (error) {
        console.error('Training start failed:', error);
        showToast('Issue In Starting Training', 'error');
        
        // Fallback to simulation
        simulateTraining();
    }
}

// Training Progress Functions
function updateTrainingProgress(progress) {
    elements.progressValue.textContent = Math.round(progress) + '%';
    
    const angle = (progress / 100) * 360;
    const progressCircle = document.querySelector('.progress-circle');
    progressCircle.style.background = `conic-gradient(var(--color-primary) ${angle}deg, var(--color-secondary) ${angle}deg)`;
    
    // Update stage progress
    elements.stageFill.style.width = progress + '%';
    elements.stageProgress.textContent = Math.round(progress) + '%';
}

function updateCurrentStage(stage, description) {
    const stageMap = {
        'uploading': 'Uploading...',
        'analyzing': 'Analyzing',
        'preprocessing': 'Data Preprocessing',
        'feature_engineering': 'FeatureE ngineering',
        'model_selection': 'Model Selection',
        'training': 'Model Training',
        'evaluation': 'Model Evaluation',
        'generating_insights': 'Generating Insights',
        'completed': 'Completed!'
    };
    
    elements.currentStage.textContent = stageMap[stage] || stage;
    elements.stageDescription.textContent = description;
}

function updateEstimatedTime(time) {
    elements.estimatedTime.textContent = time;
}

function appendLogs(logs) {
    if (!Array.isArray(logs)) logs = [logs];
    
    logs.forEach(log => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${log}`;
        elements.logEntries.appendChild(entry);
    });
    
    // Auto-scroll to bottom
    elements.logEntries.scrollTop = elements.logEntries.scrollHeight;
}

function toggleTrainingLogs() {
    elements.logsContent.classList.toggle('hidden');
    elements.toggleLogs.textContent = elements.logsContent.classList.contains('hidden') ? 'Show Details' : 'Hide Details';
}

function pauseTraining() {
    showToast('ट्रेनिंग रोकी गई', 'warning');
    // Implementation would depend on backend API
}

function cancelTraining() {
    if (confirm('क्या आप वाकई ट्रेनिंग रद्द करना चाहते हैं?')) {
        if (state.websocket) {
            state.websocket.close();
        }
        showToast('ट्रेनिंग रद्द की गई', 'error');
        goToPreviousStep();
    }
}

// Training Simulation (for demo mode)
function simulateTraining() {
    const stages = [
        { name: 'analyzing', description: 'Analyzing The Data', duration: 1500 },
        { name: 'preprocessing', description: 'Cleaning And Preparing The Data', duration: 2000 },
        { name: 'feature_engineering', description: 'Applying Feature Engineering', duration: 1800 },
        { name: 'model_selection', description: 'Selecting The Best Model', duration: 2500 },
        { name: 'training', description: 'Training The Machine Learning Model', duration: 3000 },
        { name: 'evaluation', description: 'Checking The Models Performance', duration: 1500 },
        { name: 'generating_insights', description: 'Creating Visualizations And Insights', duration: 1200 }
    ];
    
    let currentStage = 0;
    let totalProgress = 0;
    
    // Add some sample logs
    appendLogs([
        'Starting analysis pipeline...',
        'Loading dataset and initial validation...',
        'Dataset validation successful'
    ]);
    
    function updateTrainingProgress() {
        if (currentStage < stages.length) {
            const stage = stages[currentStage];
            updateCurrentStage(stage.name, stage.description);
            
            const stageProgress = 100 / stages.length;
            const targetProgress = (currentStage + 1) * stageProgress;
            
            appendLogs([`Starting ${stage.name}...`]);
            
            const progressInterval = setInterval(() => {
                totalProgress += Math.random() * 3 + 1;
                if (totalProgress >= targetProgress) {
                    totalProgress = targetProgress;
                    clearInterval(progressInterval);
                    
                    appendLogs([`${stage.name} completed successfully`]);
                    currentStage++;
                    
                    if (currentStage < stages.length) {
                        setTimeout(updateTrainingProgress, 300);
                    } else {
                        // Training complete
                        updateCurrentStage('completed', 'Analysis Completed!');
                        generateSampleResults();
                        setTimeout(() => {
                            goToNextStep();
                        }, 1000);
                    }
                }
                
                updateTrainingProgress(totalProgress);
                updateEstimatedTime(Math.max(0, Math.round((100 - totalProgress) * 2)) + ' seconds');
            }, 100);
        }
    }
    
    updateTrainingProgress();
}

function generateSampleResults() {
    state.analysisResults = {
        training_time: '4m 32s',
        best_model: state.selectedModel || 'Random Forest',
        data_quality: '94%',
        metrics: {
            accuracy: 0.94,
            precision: 0.92,
            recall: 0.95,
            f1_score: 0.93
        },
        feature_importance: {
            "feature_1": 0.35,
            "feature_2": 0.28,
            "feature_3": 0.18,
            "feature_4": 0.12,
            "feature_5": 0.07
        },
        insights: {
            key_findings: [
                'The Model Achieved 94% Accuracy',
                'Feature_1 Is The Most Important Predictor',
                'No Overfitting Observed In The Data'
            ],
            data_insights: [
                'Data Quality Is Excellent',
                'All Features Are Useful',
                'Target Variable Is Balanced'
            ],
            recommendations: [
                'This Model Can Be Deployed In Production',
                'Monitor Feature_1 In Production',
                'Focus On Feature_5 For Data'
            ]
        },
        applied_methods: ['standard scaling', 'onehot encoding', 'simple imputation'],
        hyperparameters: {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'min_samples_split': 5
        }
    };
}

// Results Display
function displayResults() {
    if (!state.analysisResults) {
        generateSampleResults();
    }
    
    const results = state.analysisResults;
    
    // Update quick stats
    elements.trainingTime.textContent = results.training_time || '4m 32s';
    elements.bestModel.textContent = results.best_model || 'Random Forest';
    elements.finalDataQuality.textContent = results.data_quality || '94%';
    
    // Update metrics
    if (results.metrics) {
        document.getElementById('accuracyMetric').textContent = (results.metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('precisionMetric').textContent = (results.metrics.precision * 100).toFixed(1) + '%';
        document.getElementById('recallMetric').textContent = (results.metrics.recall * 100).toFixed(1) + '%';
        document.getElementById('f1Metric').textContent = (results.metrics.f1_score * 100).toFixed(1) + '%';
    }
    
    // Create charts
    createFeatureImportanceChart(results.feature_importance);
    createPerformanceChart(results.metrics);
    createTrainingChart();
    createDistributionChart();
    
    // Display insights
    displayInsightsList(results.insights);
    
    // Display model details
    displayModelDetails(results);
    
    showToast('विश्लेषण परिणाम तैयार हैं!', 'success');
}

function displayInsightsList(insights) {
    if (!insights) return;
    
    // Key findings
    if (insights.key_findings && elements.keyFindings) {
        elements.keyFindings.innerHTML = '';
        insights.key_findings.forEach(finding => {
            const item = document.createElement('div');
            item.className = 'insight-item';
            item.innerHTML = `
                <div class="insight-icon">🎯</div>
                <div class="insight-text">${finding}</div>
            `;
            elements.keyFindings.appendChild(item);
        });
    }
    
    // Data insights
    if (insights.data_insights && elements.dataInsightsList) {
        elements.dataInsightsList.innerHTML = '';
        insights.data_insights.forEach(insight => {
            const item = document.createElement('div');
            item.className = 'insight-item';
            item.innerHTML = `
                <div class="insight-icon">📊</div>
                <div class="insight-text">${insight}</div>
            `;
            elements.dataInsightsList.appendChild(item);
        });
    }
    
    // Recommendations
    if (insights.recommendations && elements.recommendationsList) {
        elements.recommendationsList.innerHTML = '';
        insights.recommendations.forEach(recommendation => {
            const item = document.createElement('div');
            item.className = 'insight-item';
            item.innerHTML = `
                <div class="insight-icon">🚀</div>
                <div class="insight-text">${recommendation}</div>
            `;
            elements.recommendationsList.appendChild(item);
        });
    }
}

function displayModelDetails(results) {
    // Applied methods
    if (results.applied_methods && elements.appliedMethods) {
        elements.appliedMethods.innerHTML = '';
        results.applied_methods.forEach(method => {
            const tag = document.createElement('span');
            tag.className = 'applied-method';
            tag.textContent = method;
            elements.appliedMethods.appendChild(tag);
        });
    }
    
    // Hyperparameters
    if (results.hyperparameters && elements.hyperParameters) {
        elements.hyperParameters.innerHTML = '';
        Object.entries(results.hyperparameters).forEach(([key, value]) => {
            const param = document.createElement('div');
            param.className = 'hyperparam';
            param.innerHTML = `
                <span class="hyperparam-name">${key}</span>
                <span class="hyperparam-value">${value}</span>
            `;
            elements.hyperParameters.appendChild(param);
        });
    }
}

// Chart Functions
function createFeatureImportanceChart(featureImportance) {
    if (!featureImportance) return;
    
    const ctx = document.getElementById('featureChart').getContext('2d');
    
    if (featureChart) {
        featureChart.destroy();
    }
    
    const labels = Object.keys(featureImportance);
    const values = Object.values(featureImportance);
    
    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Importance',
                data: values,
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createPerformanceChart(metrics) {
    if (!metrics) return;
    
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    performanceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [{
                data: [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score],
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createTrainingChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    // Generate sample training progress data
    const epochs = Array.from({length: 20}, (_, i) => i + 1);
    const trainLoss = epochs.map((_, i) => 1 - (i + Math.random() * 0.1) / 25);
    const valLoss = epochs.map((_, i) => 1 - (i + Math.random() * 0.15) / 30);
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [{
                label: 'Training Loss',
                data: trainLoss,
                borderColor: '#1FB8CD',
                backgroundColor: 'rgba(31, 184, 205, 0.1)',
                tension: 0.4
            }, {
                label: 'Validation Loss',
                data: valLoss,
                borderColor: '#FFC185',
                backgroundColor: 'rgba(255, 193, 133, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createDistributionChart() {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    if (distributionChart) {
        distributionChart.destroy();
    }
    
    // Generate sample distribution data
    const labels = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'];
    const values = [23, 45, 32, 67, 18];
    
    distributionChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

// Utility Functions
function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9);
}

function populateTaskDisplayNames() {
    // This function can be used to populate task 
}

function showLoadingOverlay(text) {
    elements.loadingOverlay.classList.remove('hidden');
    elements.loadingOverlay.querySelector('.loading-text').textContent = text;
}

function hideLoadingOverlay() {
    elements.loadingOverlay.classList.add('hidden');
}

// Action Functions
function downloadResults() {
    if (!state.analysisResults) {
        showToast('No Results Found To Download', 'warning');
        return;
    }
    
    const results = {
        task: state.selectedTask,
        mode: state.selectedMode,
        execution: state.selectedExecution,
        dataset: state.selectedDataset || state.uploadedFile?.name,
        target_column: state.targetColumn,
        selected_model: state.selectedModel,
        feature_engineering: state.selectedFeatureEngineering,
        metrics: state.analysisResults.metrics,
        feature_importance: state.analysisResults.feature_importance,
        insights: state.analysisResults.insights,
        hyperparameters: state.analysisResults.hyperparameters,
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_results_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Results Successfully Downloaded', 'success');
}

function deployModel() {
    showToast('Model Deployment Feature Will Come Soon', 'info');
}

function downloadModel() {
    showToast('Model Download Feature Will Come Soon', 'info');
}

function startNewAnalysis() {
    if (confirm('Do You Want To Start A New Analysis?')) {
        // Reset state
        Object.assign(state, {
            currentStep: 1,
            selectedTask: null,
            selectedMode: null,
            selectedExecution: 'local',
            uploadedFile: null,
            selectedDataset: null,
            targetColumn: null,
            problemType: 'auto',
            validationStrategy: 'train_test_split',
            selectedFeatureEngineering: {
                encoding: [],
                scaling: [],
                imputation: [],
                selection: [],
                interactions: [],
                dimensionality: []
            },
            selectedModel: null,
            sessionId: null,
            analysisResults: null,
            trainingStartTime: null
        });
        
        // Close WebSocket
        if (state.websocket) {
            state.websocket.close();
            state.websocket = null;
        }
        
        // Reset UI
        document.querySelectorAll('.task-card, .mode-card, .dataset-card, .recommendation-card, .execution-option').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Reset forms
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        
        elements.dataPreview?.classList.add('hidden');
        elements.uploadProgress?.classList.add('hidden');
        elements.trainingSection?.classList.add('hidden');
        elements.manualSection?.classList.remove('hidden');
        
        if (elements.targetColumn) {
            elements.targetColumn.innerHTML = '<option value="">ऑटो-डिटेक्ट टारगेट कॉलम...</option>';
        }
        
        if (elements.methodsSummary) {
            elements.methodsSummary.innerHTML = '<p class="no-methods">कोई मेथड्स चुने नहीं गए हैं</p>';
        }
        
        // Destroy charts
        [featureChart, performanceChart, trainingChart, distributionChart].forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        featureChart = performanceChart = trainingChart = distributionChart = null;
        
        // Reset file input
        if (elements.fileInput) {
            elements.fileInput.value = '';
        }
        
        // Update display
        updateStepDisplay();
        updateNavigationButtons();
        updateProgressBar();
        
        showToast('New Analysis Started', 'success');
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    const titles = {
        success: 'Success',
        error: 'Error',
        warning: 'Warning',
        info: 'Information'
    };
    
    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-icon">${icons[type]}</div>
            <div class="toast-message">
                <div class="toast-title">${titles[type]}</div>
                <div class="toast-description">${message}</div>
            </div>
        </div>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

// Initialize app when DOM loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

function initializeApp() {
    console.log('🔬 Auto Data Analyst initialized');
    console.log('Backend URL:', API.baseURL);
    console.log('Features: Real-time WebSocket, Multi-language Support, Advanced Feature Engineering');
}
