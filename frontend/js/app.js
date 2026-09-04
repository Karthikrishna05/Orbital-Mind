import { ApiService } from './data.js';
import { initStarfield } from './starfield.js';

// Global Chart Instances
let trainChartInstance = null;
let rocChartInstance = null;
let qqChartInstance = null;

// Application State
const state = {
  currentRoute: 'hero',
  trainOrbit: 'GEO',
  rocMetric: 'sw',
  testData: {
    GEO: { file: null, submitted: false },
    MEO: { file: null, submitted: false }
  },
  currentTestOrbit: 'GEO',
  currentResultsOrbit: 'GEO',
  testResults: {
    GEO: null,
    MEO: null
  },
  activeResultTab: 'shapiro',
  activeQQParam: 'x_error',
  resultsDataCache: null
};

// DOM Ready initialization
document.addEventListener('DOMContentLoaded', async () => {
  // Initialize starfield canvas
  initStarfield('starfield-canvas');

  // Initialize deterministic orbit satellite animation
  initOrbitAnimation();

  // Check Backend API Connection Health
  await checkBackendConnection();

  // Setup navigation & routing
  initNavigation();

  // Setup Train Data view listeners
  initTrainDataView();

  // Setup Model Progression view listeners
  initROCView();

  // Setup Test Data view listeners
  initTestDataView();

  // Setup Results view listeners
  initResultsView();

  // Handle Resize for charts
  window.addEventListener('resize', () => {
    if (state.currentRoute === 'train-data' && trainChartInstance) {
      trainChartInstance.resize();
    } else if ((state.currentRoute === 'model-progression' || state.currentRoute === 'roc') && rocChartInstance) {
      rocChartInstance.resize();
    } else if (state.currentRoute === 'results' && state.activeResultTab === 'qq' && qqChartInstance) {
      qqChartInstance.resize();
    }
  });

  // Route initial page
  handleHashChange();
});

/**
 * Deterministic dual-orbit animation for the GEO and MEO tracks.
 */
function initOrbitAnimation() {
  const geoGroup = document.getElementById('satellite-svg-group');
  const meoGroup = document.getElementById('satellite-meo-svg-group');
  if (!geoGroup || !meoGroup) return;

  const cx = 260;
  const cy = 105;
  const geoRx = 210;
  const geoRy = 75;
  const meoRx = 145;
  const meoRy = 52;

  let geoAngle = 0;
  let meoAngle = Math.PI * 0.72;
  const geoSpeed = (2 * Math.PI) / (12 * 60);
  const meoSpeed = (2 * Math.PI) / (8 * 60);

  function animate() {
    geoAngle = (geoAngle + geoSpeed) % (2 * Math.PI);
    meoAngle = (meoAngle + meoSpeed) % (2 * Math.PI);

    const geoX = cx + geoRx * Math.cos(geoAngle);
    const geoY = cy + geoRy * Math.sin(geoAngle);
    const meoX = cx + meoRx * Math.cos(meoAngle);
    const meoY = cy + meoRy * Math.sin(meoAngle);

    geoGroup.setAttribute('transform', `translate(${geoX.toFixed(2)}, ${geoY.toFixed(2)})`);
    meoGroup.setAttribute('transform', `translate(${meoX.toFixed(2)}, ${meoY.toFixed(2)})`);

    requestAnimationFrame(animate);
  }

  animate();
}

async function checkBackendConnection() {
  const isHealthy = await ApiService.checkHealth();
  const badge = document.getElementById('api-status-badge');
  const badgeText = document.getElementById('api-status-text');

  if (badge && badgeText) {
    if (isHealthy) {
      badge.classList.add('live');
      badgeText.textContent = 'API Connected';
    } else {
      badge.classList.remove('live');
      badgeText.textContent = 'API: Standalone Mode';
    }
  }
}

/* ==========================================================================
   NAVIGATION & ROUTING
   ========================================================================== */
function initNavigation() {
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const drawerNav = document.getElementById('drawer-nav');
  const drawerBackdrop = document.getElementById('drawer-backdrop');
  const drawerLinks = document.querySelectorAll('.drawer-link');

  function toggleDrawer(open) {
    const shouldOpen = open !== undefined ? open : !drawerNav.classList.contains('open');
    drawerNav.classList.toggle('open', shouldOpen);
    drawerBackdrop.classList.toggle('open', shouldOpen);
    hamburgerBtn.classList.toggle('active', shouldOpen);
  }

  hamburgerBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleDrawer();
  });

  drawerBackdrop.addEventListener('click', () => toggleDrawer(false));

  drawerLinks.forEach((link) => {
    link.addEventListener('click', () => {
      toggleDrawer(false);
    });
  });

  window.addEventListener('hashchange', handleHashChange);
}

function handleHashChange() {
  const rawHash = window.location.hash.replace('#', '').trim();
  const hash = rawHash === 'roc' ? 'model-progression' : rawHash || 'hero';
  navigateTo(hash);
}

export function navigateTo(route) {
  const validRoutes = ['hero', 'train-data', 'model-progression', 'roc', 'test-data', 'results'];
  let targetRoute = validRoutes.includes(route) ? route : 'hero';
  if (targetRoute === 'roc') targetRoute = 'model-progression';
  state.currentRoute = targetRoute;

  if (window.location.hash !== `#${targetRoute}`) {
    window.location.hash = targetRoute;
  }

  const header = document.getElementById('global-header');
  if (targetRoute === 'hero') {
    header.classList.add('hero-mode');
  } else {
    header.classList.remove('hero-mode');
  }

  document.querySelectorAll('.drawer-link').forEach((link) => {
    const linkRoute = link.getAttribute('data-route');
    if (linkRoute === targetRoute || (targetRoute === 'model-progression' && linkRoute === 'roc')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  document.querySelectorAll('.page-view').forEach((view) => {
    view.classList.remove('active');
  });

  const targetView = document.getElementById(`view-${targetRoute}`);
  if (targetView) {
    targetView.classList.add('active');
  }

  if (targetRoute === 'train-data') {
    updateTrainView();
  } else if (targetRoute === 'model-progression') {
    // Render composition first — it has no chart dependency, so it always
    // populates even if the Chart.js CDN is unavailable.
    renderComposition();
    renderROCChart();
  } else if (targetRoute === 'results') {
    loadResultsData();
  }
}

/* ==========================================================================
   PAGE 2: TRAIN DATA
   ========================================================================== */
function initTrainDataView() {
  const geoBtn = document.getElementById('train-toggle-geo');
  const meoBtn = document.getElementById('train-toggle-meo');

  geoBtn.addEventListener('click', () => {
    if (state.trainOrbit === 'GEO') return;
    state.trainOrbit = 'GEO';
    geoBtn.classList.add('active');
    meoBtn.classList.remove('active');
    updateTrainView();
  });

  meoBtn.addEventListener('click', () => {
    if (state.trainOrbit === 'MEO') return;
    state.trainOrbit = 'MEO';
    meoBtn.classList.add('active');
    geoBtn.classList.remove('active');
    updateTrainView();
  });
}

async function updateTrainView() {
  const dataset = await ApiService.fetchTrainData(state.trainOrbit);

  document.getElementById('stat-orbit-type').textContent = dataset.orbitType;
  document.getElementById('stat-mean-eph').textContent = dataset.stats.meanEphemeris;
  document.getElementById('stat-max-eph').textContent = dataset.stats.maxEphemeris;
  document.getElementById('stat-mean-clk').textContent = dataset.stats.meanClockBias;
  document.getElementById('stat-rms-err').textContent = dataset.stats.rmsError;

  renderTrainChart(dataset);
}

function renderTrainChart(dataset) {
  const ctx = document.getElementById('train-chart');
  if (!ctx) return;

  if (trainChartInstance) {
    trainChartInstance.destroy();
  }

  // Real training dumps have many points across 7 days; thin the markers so the
  // full-window trend stays readable (representative fallback has only 7 points).
  const nPts = (dataset.timestamps || []).length;
  const dense = nPts > 30;
  const ptRadius = dense ? 0 : 4;
  const ptHover = dense ? 4 : 7;

  // @ts-ignore
  trainChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dataset.timestamps,
      datasets: [
        {
          label: 'Ephemeris Error (m)',
          data: dataset.ephemerisError,
          borderColor: '#F0F0F8',
          backgroundColor: 'rgba(240, 240, 248, 0.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.2,
          pointBackgroundColor: '#F0F0F8',
          pointBorderColor: '#000000',
          pointBorderWidth: 2,
          pointRadius: ptRadius,
          pointHoverRadius: ptHover,
          yAxisID: 'y'
        },
        {
          label: 'Clock Bias Error (m)',
          data: dataset.clockBiasError,
          borderColor: '#90D5E5',
          backgroundColor: 'rgba(144, 213, 229, 0.04)',
          fill: true,
          borderDash: [5, 4],
          tension: 0.35,
          borderWidth: 2.0,
          pointBackgroundColor: '#90D5E5',
          pointBorderColor: '#000000',
          pointBorderWidth: 2,
          pointRadius: ptRadius,
          pointHoverRadius: ptHover,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: 'easeOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#C0C0C8',
            font: { family: 'Space Grotesk', size: 12, weight: '500' },
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        tooltip: {
          backgroundColor: 'rgba(10, 10, 14, 0.95)',
          titleColor: '#F0F0F5',
          bodyColor: '#C0C0C8',
          borderColor: 'rgba(200, 200, 220, 0.3)',
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          titleFont: { family: 'Space Grotesk', size: 13, weight: '600' },
          bodyFont: { family: 'JetBrains Mono', size: 12 }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(160, 160, 175, 0.12)' },
          ticks: { color: '#A0A0AA', font: { family: 'JetBrains Mono', size: 11 } },
          title: {
            display: true,
            text: 'Training Observation Time (7-day window)',
            color: '#8A8A98',
            font: { size: 11, family: 'Space Grotesk' }
          }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          grid: { color: 'rgba(160, 160, 175, 0.12)' },
          ticks: { color: '#E8E8ED', font: { family: 'JetBrains Mono', size: 11 } },
          title: {
            display: true,
            text: 'Ephemeris Error (meters)',
            color: '#E8E8ED',
            font: { size: 11, family: 'Space Grotesk', weight: '600' }
          }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#90D5E5', font: { family: 'JetBrains Mono', size: 11 } },
          title: {
            display: true,
            text: 'Clock Bias Error (m)',
            color: '#90D5E5',
            font: { size: 11, family: 'Space Grotesk', weight: '600' }
          }
        }
      }
    }
  });
}

/* ==========================================================================
   PAGE 3: MODEL PROGRESSION
   ========================================================================== */
function initROCView() {
  // Progression is now a single Shapiro-Wilk W view (metric toggle removed);
  // nothing to wire here beyond the render triggered on navigation.
}

// Benchmark from the competition Note (target residual-normality W).
const SW_BENCHMARK = 0.981;

// Verdict per phase, derived from the authored ✅/⚠️/❌ markers already present
// in each phase's reductionPct / description (no fabricated values).
function phaseVerdict(item) {
  const s = `${item.reductionPct || ''} ${item.description || ''}`;
  if (s.includes('❌') || /reject(ed)?|spread-gamer/i.test(s)) return 'fail';
  if (s.includes('⚠️') || /partial/i.test(s)) return 'partial';
  if (s.includes('✅') || /\bpass\b/i.test(s)) return 'pass';
  return 'progress';
}

const VERDICT_COLOR = {
  pass:     { bar: 'rgba(126, 224, 160, 0.35)', border: '#7ee0a0' },
  partial:  { bar: 'rgba(224, 192, 126, 0.35)', border: '#e0c07e' },
  fail:     { bar: 'rgba(224, 138, 138, 0.32)', border: '#e08a8a' },
  progress: { bar: 'rgba(200, 200, 220, 0.20)', border: '#c8c8dc' }
};

const VERDICT_LABEL = {
  pass: 'PASS — H₀ not rejected (residuals Gaussian)',
  partial: 'PARTIAL — some channels non-normal',
  fail: 'REJECT / rejected iteration',
  progress: 'Progression step'
};

async function renderROCChart() {
  const ctx = document.getElementById('roc-chart');
  if (!ctx) return;

  const data = await ApiService.fetchROCData();
  const iterations = data.iterations;
  const labels = iterations.map((item) => item.name);
  const dataValues = iterations.map((i) => i.swW);
  const verdicts = iterations.map(phaseVerdict);
  const barColors = verdicts.map((v) => VERDICT_COLOR[v].bar);
  const barBorders = verdicts.map((v) => VERDICT_COLOR[v].border);

  if (rocChartInstance) {
    rocChartInstance.destroy();
  }

  // @ts-ignore
  rocChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          type: 'bar',
          label: 'Shapiro-Wilk W (higher = more Gaussian residuals)',
          data: dataValues,
          backgroundColor: barColors,
          borderColor: barBorders,
          borderWidth: 1.5,
          borderRadius: 5,
          barPercentage: 0.52
        },
        {
          type: 'line',
          label: `Benchmark W = ${SW_BENCHMARK.toFixed(3)}`,
          data: dataValues.map(() => SW_BENCHMARK),
          borderColor: '#90D5E5',
          borderWidth: 1.8,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          labels: {
            color: '#C0C0C8',
            font: { family: 'Space Grotesk', size: 12 }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(10, 10, 14, 0.95)',
          titleColor: '#F0F0F5',
          bodyColor: '#C0C0C8',
          borderColor: 'rgba(200, 200, 220, 0.3)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: function (context) {
              if (context.datasetIndex === 0) {
                return `W = ${Number(context.parsed.y).toFixed(3)}`;
              }
              return `Benchmark W = ${SW_BENCHMARK.toFixed(3)}`;
            },
            afterBody: function (context) {
              const idx = context[0].dataIndex;
              const item = iterations[idx];
              const v = phaseVerdict(item);
              return `\nVerdict: ${VERDICT_LABEL[v]}\nArchitecture: ${item.description}\nOutcome: ${item.reductionPct}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(160, 160, 175, 0.12)' },
          ticks: {
            color: '#A0A0AA',
            font: { family: 'Space Grotesk', size: 11 },
            maxRotation: 15
          }
        },
        y: {
          grid: { color: 'rgba(160, 160, 175, 0.12)' },
          ticks: { color: '#A0A0AA', font: { family: 'JetBrains Mono', size: 11 } },
          title: {
            display: true,
            text: 'Shapiro-Wilk W  (benchmark = 0.981; higher = more Gaussian)',
            color: '#C0C0C8',
            font: { size: 11, family: 'Space Grotesk' }
          }
        }
      }
    }
  });
}

/* Final-model composition: per channel, which sub-model produces it and the
   residual normality (W, p, H, Gaussian verdict) from the real evaluation dump. */
const COMPOSITION_ARCH = {
  GEO: {
    model: 'composite_pos_clock',
    channels: {
      x_error: 'Stacked Harmonic (p1h1 base + p1h3 residual)',
      y_error: 'Stacked Harmonic (p1h1 base + p1h3 residual)',
      z_error: 'Stacked Harmonic (p1h1 base + p1h3 residual)',
      satclockerror: 'Segmented Clock (change-point Kalman)'
    }
  },
  MEO1: {
    model: 'ensemble_median',
    channels: {
      x_error: 'Ensemble Median (harmonic-p2h2 · auto-harmonic-np3h2 · GBR-Huber)',
      y_error: 'Ensemble Median (harmonic-p2h2 · auto-harmonic-np3h2 · GBR-Huber)',
      z_error: 'Ensemble Median (harmonic-p2h2 · auto-harmonic-np3h2 · GBR-Huber)',
      satclockerror: 'Ensemble Median (harmonic-p2h2 · auto-harmonic-np3h2 · GBR-Huber)'
    }
  },
  MEO2: {
    model: 'composite_pos_clock',
    channels: {
      x_error: 'Stacked Harmonic (p1h1 base + p1h3 residual)',
      y_error: 'Stacked Harmonic (p1h1 base + p1h3 residual)',
      z_error: 'Stacked Harmonic (p1h1 base + p1h3 residual)',
      satclockerror: 'Segmented Clock (change-point Kalman)'
    }
  }
};

const CHANNEL_LABEL = {
  x_error: 'x_error (Radial)',
  y_error: 'y_error (Along-Track)',
  z_error: 'z_error (Cross-Track)',
  satclockerror: 'satclockerror (Clock)'
};

function renderComposition() {
  const container = document.getElementById('composition-container');
  if (!container) return;

  const dyn = window.DYNAMIC_GNSS_RESULTS;
  if (!dyn) {
    container.innerHTML = `<p style="color: var(--silver-secondary); font-size: 0.82rem;">
      Per-channel normality unavailable — run <code>python scripts/evaluate_day8.py</code> to generate it.</p>`;
    return;
  }

  const orbits = ['GEO', 'MEO1', 'MEO2'].filter((o) => dyn[o]);
  container.innerHTML = orbits.map((orbit) => {
    const arch = COMPOSITION_ARCH[orbit];
    const chans = dyn[orbit].channels || {};
    const rows = Object.keys(CHANNEL_LABEL).map((param) => {
      const c = chans[param];
      if (!c) return '';
      const gaussian = c.H === 0
        ? '<span class="gaussian-yes">Gaussian ✅</span>'
        : '<span class="gaussian-no">Non-Gaussian ❌</span>';
      return `
        <tr>
          <td style="font-weight:600;">${CHANNEL_LABEL[param]}</td>
          <td style="color: var(--silver-secondary); font-size: 0.78rem;">${arch.channels[param]}</td>
          <td>${c.W.toFixed(4)}</td>
          <td>${c.p.toFixed(4)}</td>
          <td>${c.H}</td>
          <td>${gaussian}</td>
        </tr>`;
    }).join('');

    const agg = dyn[orbit].aggregate;
    const aggGauss = agg.H === 0
      ? '<span class="gaussian-yes">Gaussian ✅</span>'
      : (agg.H < 1 ? '<span style="color:#e0c07e;font-weight:600;">Partial ⚠️</span>'
                   : '<span class="gaussian-no">Non-Gaussian ❌</span>');

    return `
      <div class="composition-group">
        <div class="composition-group-title">${orbit}</div>
        <div class="composition-group-model">Final model: <code>${arch.model}</code></div>
        <div class="table-responsive">
          <table class="gnss-table">
            <thead>
              <tr>
                <th>Channel</th>
                <th>Sub-model (component)</th>
                <th>W</th>
                <th>p</th>
                <th>H</th>
                <th>Gaussian nature</th>
              </tr>
            </thead>
            <tbody>
              ${rows}
              <tr class="aggregate-row">
                <td style="font-weight:700;">Aggregate (4-channel)</td>
                <td style="color: var(--silver-secondary); font-size: 0.78rem;">composite</td>
                <td>${agg.W.toFixed(4)}</td>
                <td>${agg.p.toFixed(4)}</td>
                <td>${agg.H.toFixed(2)}</td>
                <td>${aggGauss}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>`;
  }).join('');
}

/* ==========================================================================
   PAGE 4: TEST DATA
   ========================================================================== */
function initTestDataView() {
  const geoToggle = document.getElementById('test-toggle-geo');
  const meoToggle = document.getElementById('test-toggle-meo');
  const dropzone = document.getElementById('test-dropzone');
  const fileInput = document.getElementById('test-file-input');
  const submitBtn = document.getElementById('test-submit-btn');
  const fileStatusCard = document.getElementById('test-file-status');
  const fileNameText = document.getElementById('test-file-name');
  const fileMetaText = document.getElementById('test-file-meta');
  const removeFileBtn = document.getElementById('btn-remove-file');
  const submissionAlert = document.getElementById('test-submission-alert');
  const testDesc = document.getElementById('test-description-text');

  function updateTestOrbitContext(orbit) {
    state.currentTestOrbit = orbit;
    if (orbit === 'GEO') {
      geoToggle.classList.add('active');
      meoToggle.classList.remove('active');
      testDesc.innerHTML = 'Insert <strong>GEO</strong> test data (ephemeris & clock telemetry):';
    } else {
      meoToggle.classList.add('active');
      geoToggle.classList.remove('active');
      testDesc.innerHTML = 'Insert <strong>MEO</strong> test data (ephemeris & clock telemetry):';
    }

    renderTestUploadState();
    submissionAlert.classList.remove('visible');
  }

  geoToggle.addEventListener('click', () => {
    if (state.currentTestOrbit === 'GEO') return;
    updateTestOrbitContext('GEO');
  });

  meoToggle.addEventListener('click', () => {
    if (state.currentTestOrbit === 'MEO') return;
    updateTestOrbitContext('MEO');
  });

  dropzone.addEventListener('click', () => {
    fileInput.value = '';
    fileInput.click();
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    // @ts-ignore
    if (e.target.files && e.target.files.length > 0) {
      // @ts-ignore
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    state.testData[state.currentTestOrbit].file = file;
    state.testData[state.currentTestOrbit].submitted = false;
    renderTestUploadState();
  }

  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.testData[state.currentTestOrbit].file = null;
    state.testData[state.currentTestOrbit].submitted = false;
    fileInput.value = '';
    renderTestUploadState();
  });

  submitBtn.addEventListener('click', async () => {
    const currentOrbit = state.currentTestOrbit;
    const file = state.testData[currentOrbit].file;
    if (!file) return;

    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <svg class="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="12"></circle>
      </svg>
      Processing Telemetry...
    `;

    try {
      const result = await ApiService.uploadTestData(file, currentOrbit);
      state.testResults[currentOrbit] = result;
      state.testData[currentOrbit].submitted = true;
      state.currentResultsOrbit = currentOrbit;

      submitBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        Inference Complete
      `;
      submissionAlert.innerHTML = `
        🚀 <strong>${currentOrbit} Inference Successful:</strong> File <code>${file.name}</code> processed against the
        ${result.model} model (n=${result.n} points).
        Residual distributions and normality metrics are updated in the <a href="#results" style="color: var(--silver-bright); text-decoration: underline;">Results</a> panel.
      `;
      submissionAlert.classList.add('visible');

      navigateTo('results');
    } catch (err) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `Submit ${currentOrbit} Test Data`;
      submissionAlert.innerHTML = `
        ⚠️ <strong>${currentOrbit} Inference Failed:</strong> ${err.message}
      `;
      submissionAlert.classList.add('visible');
    }
  });

  function renderTestUploadState() {
    const currentOrbit = state.currentTestOrbit;
    const file = state.testData[currentOrbit].file;

    if (file) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `Submit ${currentOrbit} Test Data`;
      fileNameText.textContent = file.name;
      const sizeKb = (file.size / 1024).toFixed(1);
      fileMetaText.textContent = `Size: ${sizeKb} KB • Context: ${currentOrbit} Orbit • Ready for submission`;
      fileStatusCard.classList.add('visible');
    } else {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `Submit ${currentOrbit} Test Data`;
      fileStatusCard.classList.remove('visible');
      submissionAlert.classList.remove('visible');
    }
  }
}

/* ==========================================================================
   PAGE 5: RESULTS
   ========================================================================== */
function initResultsView() {
  const orbitGeoToggle = document.getElementById('res-toggle-geo');
  const orbitMeoToggle = document.getElementById('res-toggle-meo');

  const tabShapiro = document.getElementById('res-tab-shapiro');
  const tabMeanSD = document.getElementById('res-tab-meansd');
  const tabQQ = document.getElementById('res-tab-qq');
  const testAgainBtn = document.getElementById('results-test-again');
  const homeBtn = document.getElementById('results-home');

  const panelShapiro = document.getElementById('panel-shapiro');
  const panelMeanSD = document.getElementById('panel-meansd');
  const panelQQ = document.getElementById('panel-qq');

  function switchResultsOrbit(orbit) {
    if (state.currentResultsOrbit === orbit) return;
    state.currentResultsOrbit = orbit;
    orbitGeoToggle.classList.toggle('active', orbit === 'GEO');
    orbitMeoToggle.classList.toggle('active', orbit === 'MEO');
    renderResultsForCurrentOrbit();
  }

  orbitGeoToggle.addEventListener('click', () => switchResultsOrbit('GEO'));
  orbitMeoToggle.addEventListener('click', () => switchResultsOrbit('MEO'));

  tabShapiro.addEventListener('click', () => switchResultsTab('shapiro'));
  tabMeanSD.addEventListener('click', () => switchResultsTab('meansd'));
  tabQQ.addEventListener('click', () => switchResultsTab('qq'));
  testAgainBtn.addEventListener('click', () => navigateTo('test-data'));
  homeBtn.addEventListener('click', () => navigateTo('hero'));

  const qqBtns = document.querySelectorAll('.qq-param-btn');
  qqBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const param = btn.getAttribute('data-param');
      state.activeQQParam = param;
      qqBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderQQChart(param);
    });
  });

  function switchResultsTab(tab) {
    state.activeResultTab = tab;
    [tabShapiro, tabMeanSD, tabQQ].forEach((t) => t.classList.remove('active'));
    [panelShapiro, panelMeanSD, panelQQ].forEach((p) => p.classList.remove('active'));

    if (tab === 'shapiro') {
      tabShapiro.classList.add('active');
      panelShapiro.classList.add('active');
    } else if (tab === 'meansd') {
      tabMeanSD.classList.add('active');
      panelMeanSD.classList.add('active');
    } else if (tab === 'qq') {
      tabQQ.classList.add('active');
      panelQQ.classList.add('active');
      renderQQChart(state.activeQQParam);
    }
  }
}

async function loadResultsData() {
  if (!state.resultsDataCache) {
    state.resultsDataCache = await ApiService.fetchResultsData();
  }
  renderResultsForCurrentOrbit();
}

function renderResultsForCurrentOrbit() {
  const orbit = state.currentResultsOrbit;
  const live = state.testResults[orbit];

  if (live) {
    populateShapiroTable(live.shapiroWilk);
    populateMeanSDTable(live.meanAndSD);
  } else if (state.resultsDataCache) {
    const filterByOrbit = (rows = []) =>
      rows.filter((row) => row.parameter.includes(orbit));
    populateShapiroTable(filterByOrbit(state.resultsDataCache.shapiroWilk));
    populateMeanSDTable(filterByOrbit(state.resultsDataCache.meanAndSD));
  }

  if (state.activeResultTab === 'qq') {
    renderQQChart(state.activeQQParam);
  }
}

function getActiveQQPlotData() {
  const orbit = state.currentResultsOrbit;
  const live = state.testResults[orbit];
  if (live) return live.qqPlotData;
  return state.resultsDataCache?.qqPlotData?.[orbit];
}

function populateShapiroTable(rows = []) {
  const tbody = document.getElementById('shapiro-table-body');
  if (!tbody || !rows) return;

  tbody.innerHTML = rows
    .map((row, idx) => {
      const isAgg = idx === rows.length - 1;
      // Map actual status to badge class
      const badgeClass = row.status === 'normal'
        ? 'badge-normal'
        : row.status === 'partial'
          ? 'badge-partial'
          : 'badge-non-normal';
      return `
        <tr class="${isAgg ? 'aggregate-row' : ''}">
          <td style="font-weight: ${isAgg ? '700' : '500'};">${row.parameter}</td>
          <td>${row.wStatistic}</td>
          <td>${row.pValue}</td>
          <td><span class="${badgeClass}">${row.hypothesis}</span></td>
          <td style="color: var(--silver-secondary); font-size: 0.78rem;">${row.notes}</td>
        </tr>
      `;
    })
    .join('');
}

function populateMeanSDTable(rows = []) {
  const tbody = document.getElementById('meansd-table-body');
  if (!tbody || !rows) return;

  tbody.innerHTML = rows
    .map((row, idx) => {
      const isAgg = idx === rows.length - 1;
      return `
        <tr class="${isAgg ? 'aggregate-row' : ''}">
          <td style="font-weight: ${isAgg ? '700' : '500'};">${row.parameter}</td>
          <td style="color: var(--silver-secondary);">${row.unit}</td>
          <td style="color: var(--silver-bright); font-weight: 600;">${row.mean}</td>
          <td style="color: var(--silver-primary); font-weight: 600;">${row.sd}</td>
          <td>${row.confidence95}</td>
          <td>${row.maxResidual}</td>
        </tr>
      `;
    })
    .join('');
}

function renderQQChart(param = 'x_error') {
  const ctx = document.getElementById('qq-chart');
  if (!ctx) return;

  const qqData = getActiveQQPlotData()?.[param];
  if (!qqData) return;

  const theoreticals = qqData.points.map((p) => p.theoretical);
  const minZ = Math.min(...theoreticals);
  const maxZ = Math.max(...theoreticals);

  const samples = qqData.points.map((p) => p.sample);
  const minSample = Math.min(...samples);
  const maxSample = Math.max(...samples);

  const scatterPoints = qqData.points.map((p) => ({
    x: p.theoretical,
    y: p.sample
  }));

  const refLinePoints = [
    { x: minZ, y: minSample },
    { x: maxZ, y: maxSample }
  ];

  if (qqChartInstance) {
    qqChartInstance.destroy();
  }

  // @ts-ignore
  qqChartInstance = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Residual Quantiles',
          data: scatterPoints,
          backgroundColor: '#F0F0F8',
          borderColor: '#000000',
          borderWidth: 1,
          pointRadius: 4.5,
          pointHoverRadius: 7
        },
        {
          type: 'line',
          label: 'Theoretical Normal Fit Line (45°)',
          data: refLinePoints,
          borderColor: '#90D5E5',
          borderWidth: 1.8,
          borderDash: [5, 4],
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500 },
      plugins: {
        legend: {
          labels: {
            color: '#C0C0C8',
            font: { family: 'Space Grotesk', size: 11 }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(10, 10, 14, 0.95)',
          titleColor: '#F0F0F5',
          bodyColor: '#C0C0C8',
          borderColor: 'rgba(200, 200, 220, 0.3)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: function (ctx) {
              return `Theoretical: ${ctx.parsed.x} σ | Sample Residual: ${ctx.parsed.y} ${qqData.unit}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          grid: { color: 'rgba(160, 160, 175, 0.12)' },
          ticks: { color: '#A0A0AA', font: { family: 'JetBrains Mono', size: 11 } },
          title: {
            display: true,
            text: 'Theoretical Standard Normal Quantiles (Z-Score)',
            color: '#C0C0C8',
            font: { size: 11, family: 'Space Grotesk' }
          }
        },
        y: {
          type: 'linear',
          grid: { color: 'rgba(160, 160, 175, 0.12)' },
          ticks: { color: '#A0A0AA', font: { family: 'JetBrains Mono', size: 11 } },
          title: {
            display: true,
            text: `Sample Residual Quantiles (${qqData.unit})`,
            color: '#C0C0C8',
            font: { size: 11, family: 'Space Grotesk' }
          }
        }
      }
    }
  });
}
