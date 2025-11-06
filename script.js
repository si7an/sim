// Show/hide sections
function showSection(sectionName) {
    document.querySelectorAll('.content-section').forEach(section => {
        section.style.display = 'none';
    });
    
    if (sectionName === 'predict') {
        document.getElementById('predict-section').style.display = 'block';
    } else if (sectionName === 'history') {
        document.getElementById('history-section').style.display = 'block';
        loadHistory();
    }
}

// Handle prediction form submission
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const submitButton = this.querySelector('button[type="submit"]');
    
    try {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResult(result);
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-calculator"></i> Predict Sepsis Risk';
    }
});

// Display prediction result
function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const riskClass = result.risk_level.toLowerCase().includes('high') ? 'risk-high' :
                     result.risk_level.toLowerCase().includes('medium') ? 'risk-medium' : 'risk-low';
    
    resultDiv.innerHTML = `
        <div class="alert ${riskClass}">
            <h4><i class="fas fa-${result.prediction === 'Positive' ? 'exclamation-triangle' : 'check-circle'}"></i> 
                Prediction: ${result.prediction}</h4>
            <p><strong>Probability:</strong> ${result.probability}%</p>
            <p><strong>Risk Level:</strong> ${result.risk_level}</p>
            <p><strong>Recommendation:</strong> ${getRecommendation(result)}</p>
        </div>
    `;
    resultDiv.style.display = 'block';
}

// Get recommendation based on risk level
function getRecommendation(result) {
    if (result.risk_level === 'High Risk') {
        return 'Immediate medical attention required. Consider ICU transfer and broad-spectrum antibiotics.';
    } else if (result.risk_level === 'Medium Risk') {
        return 'Close monitoring recommended. Repeat labs in 4-6 hours. Consider antibiotic therapy.';
    } else {
        return 'Continue routine monitoring. Low suspicion for sepsis.';
    }
}

// Train model
async function trainModel() {
    try {
        const response = await fetch('/train', { method: 'POST' });
        const result = await response.json();
        
        if (response.ok) {
            alert(result.message);
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Load prediction history
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        const historyContent = document.getElementById('history-content');
        
        if (history.length === 0) {
            historyContent.innerHTML = '<p>No prediction history available.</p>';
            return;
        }
        
        historyContent.innerHTML = history.map((item, index) => `
            <div class="history-item">
                <h6>Prediction ${history.length - index}</h6>
                <p><strong>Result:</strong> ${item.result.prediction} (${item.result.probability}%) - ${item.result.risk_level}</p>
                <small class="text-muted">Age: ${item.data[0]}, HR: ${item.data[1]}, Temp: ${item.data[3]}°C</small>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Clear history
async function clearHistory() {
    if (confirm('Are you sure you want to clear all prediction history?')) {
        try {
            await fetch('/clear_history', { method: 'POST' });
            loadHistory();
        } catch (error) {
            alert('Error clearing history: ' + error.message);
        }
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    showSection('predict');
});