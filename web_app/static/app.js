const modelSelect = document.getElementById('modelSelect');
const modelInfo = document.getElementById('modelInfo');
const featureInput = document.getElementById('featureInput');
const resultBox = document.getElementById('resultBox');
const fillBtn = document.getElementById('fillBtn');
const predictBtn = document.getElementById('predictBtn');

let models = [];

function randomVector(n) {
  return Array.from({ length: n }, () => (Math.random() * 1.0).toFixed(6)).join(', ');
}

function currentModel() {
  return models.find(m => m.name === modelSelect.value);
}

function refreshModelInfo() {
  const m = currentModel();
  if (!m) return;
  modelInfo.textContent = `Expected features: ${m.input_dim} | Classes: ${m.num_labels}`;
}

async function loadModels() {
  const res = await fetch('/api/models');
  const data = await res.json();
  models = data.models;
  modelSelect.innerHTML = models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
  refreshModelInfo();
  if (!data.torch_available) {
    resultBox.textContent = `⚠ PyTorch unavailable in this environment. Web UI still works in mock mode.\n${data.torch_error || ''}`;
  }
}

modelSelect.addEventListener('change', refreshModelInfo);

fillBtn.addEventListener('click', () => {
  const m = currentModel();
  if (!m) return;
  featureInput.value = randomVector(m.input_dim);
});

predictBtn.addEventListener('click', async () => {
  const model = modelSelect.value;
  const features = featureInput.value.trim();
  resultBox.textContent = 'Running...';
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, features })
  });
  const data = await res.json();
  resultBox.textContent = JSON.stringify(data, null, 2);
});

loadModels();
