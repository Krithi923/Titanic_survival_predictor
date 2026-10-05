// ⚠️ Backend URL — change this after deployment to Render
const API_URL = "http://127.0.0.1:5000";

// ---- Element references ----
const form        = document.getElementById("predictorForm");
const submitBtn   = document.getElementById("submitBtn");
const btnText     = submitBtn.querySelector(".btn-text");
const btnLoader   = submitBtn.querySelector(".btn-loader");
const errorMsg    = document.getElementById("errorMsg");

const idleState    = document.getElementById("idleState");
const loadingState = document.getElementById("loadingState");
const resultState  = document.getElementById("resultState");

const resultIcon  = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const probNumber  = document.getElementById("probNumber");
const gaugeFill   = document.getElementById("gaugeFill");
const apiStatus   = document.getElementById("apiStatus");

// ---- Backend health check on page load ----
fetch(`${API_URL}/`)
    .then(r => r.json())
    .then(() => { apiStatus.textContent = "Connected"; })
    .catch(() => { apiStatus.textContent = "Offline"; });

// ---- Form submit ----
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorMsg.classList.add("hidden");

    // Gather inputs
    const payload = {
        pclass: parseInt(document.getElementById("pclass").value),
        sex:    document.getElementById("sex").value,
        age:    parseFloat(document.getElementById("age").value),
        sibsp:  parseInt(document.getElementById("sibsp").value),
        parch:  parseInt(document.getElementById("parch").value),
    };

    // Show loading state
    setState("loading");
    submitBtn.disabled = true;
    btnText.textContent = "Predicting…";
    btnLoader.classList.remove("hidden");

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!data.success) throw new Error(data.error || "Prediction failed");

        showResult(data.survived, data.probability);

    } catch (err) {
        errorMsg.textContent = `⚠️ ${err.message}`;
        errorMsg.classList.remove("hidden");
        setState("idle");
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = "Predict Survival";
        btnLoader.classList.add("hidden");
    }
});

// ---- State switcher ----
function setState(state) {
    idleState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.add("hidden");

    if (state === "idle")    idleState.classList.remove("hidden");
    if (state === "loading") loadingState.classList.remove("hidden");
    if (state === "result")  resultState.classList.remove("hidden");
}

// ---- Show prediction result ----
function showResult(survived, probability) {
    setState("result");

    if (survived) {
        resultIcon.textContent = "🛟";
        resultTitle.textContent = "Survived";
        resultTitle.className = "survived";
    } else {
        resultIcon.textContent = "🌊";
        resultTitle.textContent = "Deceased";
        resultTitle.className = "deceased";
    }

    probNumber.textContent = probability.toFixed(1);

    // Reset + animate gauge
    gaugeFill.style.width = "0%";
    setTimeout(() => {
        gaugeFill.style.width = `${probability}%`;
    }, 50);

    // Gauge color depends on outcome
    gaugeFill.style.background = survived
        ? "linear-gradient(90deg, #38bdf8, #10b981)"
        : "linear-gradient(90deg, #f59e0b, #ef4444)";
}