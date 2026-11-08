(() => {
  const form = document.querySelector("#upload-form");
  const input = document.querySelector("#audio-file");
  const dropZone = document.querySelector("#drop-zone");
  const selectedFile = document.querySelector("#selected-file");
  const fileName = document.querySelector("#file-name");
  const removeButton = document.querySelector("#remove-file");
  const analyzeButton = document.querySelector("#analyze-button");
  const buttonLabel = document.querySelector(".button-label");
  const spinner = document.querySelector(".spinner");
  const errorMessage = document.querySelector("#error-message");
  const resultCard = document.querySelector("#result-card");
  const resultIcon = document.querySelector("#result-icon");
  const resultGender = document.querySelector("#result-gender");
  const confidenceValue = document.querySelector("#confidence-value");
  const confidenceBar = document.querySelector("#confidence-bar");
  const tryAnother = document.querySelector("#try-another");
  const maxBytes = 16 * 1024 * 1024;
  let file = null;

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  }

  function chooseFile(candidate) {
    errorMessage.hidden = true;
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".wav")) {
      showError("Please choose a .wav audio file.");
      return;
    }
    if (candidate.size > maxBytes) {
      showError("This file exceeds the 16 MB upload limit.");
      return;
    }
    file = candidate;
    fileName.textContent = candidate.name;
    selectedFile.hidden = false;
    analyzeButton.disabled = false;
  }

  function reset() {
    file = null;
    input.value = "";
    selectedFile.hidden = true;
    resultCard.hidden = true;
    resultCard.classList.remove("is-male", "is-female");
    confidenceBar.style.width = "0";
    analyzeButton.disabled = true;
    errorMessage.hidden = true;
    form.hidden = false;
  }

  dropZone.addEventListener("click", (event) => {
    if (event.target.closest("#remove-file")) return;
    input.click();
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.target === dropZone && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => chooseFile(input.files[0]));
  removeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    reset();
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!file) return;
    errorMessage.hidden = true;
    analyzeButton.disabled = true;
    buttonLabel.textContent = "Analyzing...";
    spinner.hidden = false;
    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch("/predict", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Analysis failed. Please try again.");
      const gender = data.gender.toLowerCase();
      if (!["male", "female"].includes(gender)) throw new Error("The server returned an invalid result.");
      const confidence = Math.max(0, Math.min(1, Number(data.confidence)));
      const percent = Math.round(confidence * 100);
      resultCard.classList.add(`is-${gender}`);
      resultIcon.textContent = gender === "male" ? "👨" : "👩";
      resultGender.textContent = gender.toUpperCase();
      confidenceValue.textContent = `${percent}%`;
      confidenceBar.style.width = `${percent}%`;
      resultCard.querySelector('[role="progressbar"]').setAttribute("aria-valuenow", String(percent));
      form.hidden = true;
      resultCard.hidden = false;
    } catch (error) {
      showError(error.message || "Unable to analyze this file. Please try again.");
      analyzeButton.disabled = false;
    } finally {
      buttonLabel.textContent = "Analyze Voice";
      spinner.hidden = true;
    }
  });

  tryAnother.addEventListener("click", reset);
})();
