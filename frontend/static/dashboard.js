// --- Show More Functionality ---
let start = 10;
const batchSize = 10;
const total = healthScores.length;
const tableBody = document.querySelector("#predTable tbody");
const showMoreBtn = document.getElementById("showMoreBtn");

if (showMoreBtn) {
  showMoreBtn.addEventListener("click", () => {
    const end = Math.min(start + batchSize, total);
    for (let i = start; i < end; i++) {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${i + 1}</td>
                       <td>${healthScores[i].toFixed(2)}</td>
                       <td>${chdProbs[i].toFixed(2)}</td>`;
      tableBody.appendChild(row);
    }
    start += batchSize;
    if (start >= total) showMoreBtn.style.display = "none";
  });
}

// --- Chart.js Visuals ---
const ctxHealth = document.getElementById('healthScoreChart').getContext('2d');
new Chart(ctxHealth, {
  type: 'bar',
  data: {
    labels: healthScores.map((_, i) => i + 1),
    datasets: [{
      label: 'Health Score',
      data: healthScores,
      backgroundColor: 'rgba(25, 135, 84, 0.7)',
      borderColor: 'rgba(25, 135, 84, 1)',
      borderWidth: 1
    }]
  },
  options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
});

const ctxCHD = document.getElementById('chdProbChart').getContext('2d');
new Chart(ctxCHD, {
  type: 'bar',
  data: {
    labels: chdProbs.map((_, i) => i + 1),
    datasets: [{
      label: 'CHD Probability (%)',
      // ✅ Convert 0–1 to 0–100
      data: chdProbs.map(v => v * 100),
      backgroundColor: 'rgba(220, 53, 69, 0.7)',
      borderColor: 'rgba(220, 53, 69, 1)',
      borderWidth: 1
    }]
  },
  options: {
    responsive: true,
    scales: {
      y: { beginAtZero: true, max: 100 }
    },
    plugins: {
      title: {
        display: true,
        text: 'CHD Probability Distribution',
        font: { size: 18, weight: 'bold' },
        color: '#222'
      }
    }
  }
});

// --- Cluster Distribution Pie Chart ---
const ctxCluster = document.getElementById('clusterChart').getContext('2d');
new Chart(ctxCluster, {
  type: 'pie',
  data: {
    labels: clusterLabels,
    datasets: [{
      data: clusterValues,
      backgroundColor: [
        'rgba(25, 135, 84, 0.8)',  // low risk
        'rgba(255, 193, 7, 0.8)',  // moderate risk
        'rgba(220, 53, 69, 0.8)'   // high risk
      ],
      borderColor: '#fff',
      borderWidth: 1
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { position: 'bottom' },
      tooltip: {
        callbacks: {
          label: function(ctx) {
            return `${ctx.label}: ${ctx.raw}%`;
          }
        }
      }
    }
  }
});