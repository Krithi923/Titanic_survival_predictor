document.getElementById('predictorForm').addEventListener('submit', function(e) {
    e.preventDefault();

    // 1. Get input values
    const pclass = parseInt(document.getElementById('pclass').value);
    const sex = document.getElementById('sex').value;
    const age = parseFloat(document.getElementById('age').value);
    const sibsp = parseInt(document.getElementById('sibsp').value);
    const parch = parseInt(document.getElementById('parch').value);

    // 2. Map Categorical Variables to Numbers
    const isMale = (sex === 'male') ? 1 : 0;

    // 3. Logistic Regression Coefficients (Approximate weights trained from the actual Titanic dataset)
    const intercept = 2.50;
    const w_pclass = -1.20;  // Higher class index reduces survival chances
    const w_male = -2.70;    // Being male reduces survival chances drastically
    const w_age = -0.04;     // Higher age reduces survival chances slightly
    const w_sibsp = -0.35;   // Having too many siblings reduces survival chances
    const w_parch = -0.10;   // Family size impact

    // 4. Calculate Log-Odds (z)
    let z = intercept + 
            (w_pclass * pclass) + 
            (w_male * isMale) + 
            (w_age * age) + 
            (w_sibsp * sibsp) + 
            (w_parch * parch);

    // 5. Apply Sigmoid Function to find Probability (0 to 1)
    let probability = 1 / (1 + Math.exp(-z));
    let survivalPercent = (probability * 100).toFixed(1);

    // 6. UI Updates
    const resultBox = document.getElementById('result');
    const predictionText = document.getElementById('predictionText');
    const progressBar = document.getElementById('progressBar');

    resultBox.classList.remove('hidden');
    
    if (survivalPercent >= 50) {
        predictionText.innerHTML = `<strong>Survived!</strong> <br> This passenger has a <strong>${survivalPercent}%</strong> chance of survival.`;
        progressBar.style.backgroundColor = '#28a745'; // Green
    } else {
        predictionText.innerHTML = `<strong>Did Not Survive.</strong> <br> This passenger has only a <strong>${survivalPercent}%</strong> chance of survival.`;
        progressBar.style.backgroundColor = '#dc3545'; // Red
    }

    progressBar.style.width = survivalPercent + '%';
});// script.js
document.getElementById('predictorForm').addEventListener('submit', function(e) {
    e.preventDefault();

    // 1. Gather User inputs
    const pclass = parseInt(document.getElementById('pclass').value);
    const sex = document.getElementById('sex').value;
    const age = parseFloat(document.getElementById('age').value);
    const sibsp = parseInt(document.getElementById('sibsp').value);
    const parch = parseInt(document.getElementById('parch').value);

    // 2. Exact pipeline encoding match: female = 0, male = 1
    const isMale = (sex === 'male') ? 1 : 0;

    // 3. Mathematical Parameters extracted from Section 4 Model training
    const intercept = 4.8872;
    const w_pclass = -1.1895;
    const w_male = -2.6052;
    const w_age = -0.0305;
    const w_sibsp = -0.3547;
    const w_parch = -0.0784;

    // 4. Compute Log-Odds Equation: z = β0 + β1X1 + β2X2 + ... + βnXn
    let z = intercept + 
            (w_pclass * pclass) + 
            (w_male * isMale) + 
            (w_age * age) + 
            (w_sibsp * sibsp) + 
            (w_parch * parch);

    // 5. Pass through the Sigmoid activation function to get a probability value [0, 1]
    let probability = 1 / (1 + Math.exp(-z));
    let survivalPercent = (probability * 100).toFixed(1);

    // 6. Output UI Display logic
    const resultBox = document.getElementById('result');
    const predictionText = document.getElementById('predictionText');
    const progressBar = document.getElementById('progressBar');

    resultBox.classList.remove('hidden');
    
    if (survivalPercent >= 50) {
        predictionText.innerHTML = `<strong>Prediction: Survived</strong><br>Calculated Survival Probability: <strong>${survivalPercent}%</strong>`;
        progressBar.style.backgroundColor = '#28a745';
    } else {
        predictionText.innerHTML = `<strong>Prediction: Deceased</strong><br>Calculated Survival Probability: <strong>${survivalPercent}%</strong>`;
        progressBar.style.backgroundColor = '#dc3545';
    }

    progressBar.style.width = survivalPercent + '%';
});