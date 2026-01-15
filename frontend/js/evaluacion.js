document.addEventListener("DOMContentLoaded", async () => {
    const datasetId = sessionStorage.getItem("dataset_id");
    if (!datasetId) {
        alert("No hay dataset cargado");
        return;
    }

    const progressText = document.getElementById("progress-text");
    const progressBar = document.getElementById("progress-bar");

    try {
        /* ===============================
           1. INICIAR EVALUACIÓN
        =============================== */
        progressText.textContent = "Iniciando evaluación del modelo...";

        const startResponse = await fetch(`${API_BASE_URL}/apis/evaluar_modelo/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_id: datasetId })
        });

        if (!startResponse.ok) {
            throw new Error("No se pudo iniciar la evaluación");
        }

        const { task_id } = await startResponse.json();

        /* ===============================
           2. POLLING
        =============================== */
        while (true) {
            const estadoRes = await fetch(
                `${API_BASE_URL}/apis/estado_modelo/${task_id}/`
            );

            if (!estadoRes.ok) {
                throw new Error("Error consultando estado");
            }

            const estado = await estadoRes.json();

            progressBar.style.width = `${estado.progress || 0}%`;
            progressText.textContent = estado.message || "Procesando...";

            if (estado.status === "done") {
                progressBar.style.width = "100%";
                progressText.textContent = "Evaluación completada";
                renderResultados(estado.result);
                break;
            }

            if (estado.status === "error") {
                throw new Error(estado.error);
            }

            await new Promise(r => setTimeout(r, 1500));
        }

    } catch (err) {
        console.error(err);
        progressText.textContent = "Error durante la evaluación";
        alert(err.message);
    }
});

/* ===============================
   RENDER DE RESULTADOS
=============================== */
function renderResultados(data) {
    const m = data.metrics;

    // ===============================
    // Accuracy
    // ===============================
    document.getElementById("acc-val").textContent =
        m.accuracy_validation.toFixed(4);

    document.getElementById("acc-test").textContent =
        m.accuracy_test.toFixed(4);

    document.getElementById("acc-diff").textContent =
        m.accuracy_difference.toFixed(4);

    // ===============================
    // F1 / Precision / Recall / ROC
    // ===============================
    document.getElementById("f1-score").textContent =
        m.f1_score.toFixed(4);

    document.getElementById("precision").textContent =
        m.precision.toFixed(4);

    document.getElementById("recall").textContent =
        m.recall.toFixed(4);

    document.getElementById("roc-auc").textContent =
        m.roc_auc.toFixed(4);

    // ===============================
    // Interpretación
    // ===============================
    const diff = m.accuracy_difference;
    const tbody = document.querySelector("#interpretation-table tbody");

    let cls, text, state;
    if (diff < 0.02) {
        cls = "ok"; text = "Excelente generalización"; state = "Óptimo";
    } else if (diff < 0.05) {
        cls = "ok"; text = "Generalización normal"; state = "Aceptable";
    } else if (diff < 0.10) {
        cls = "warn"; text = "Posible overfitting"; state = "Riesgo";
    } else {
        cls = "bad"; text = "Overfitting severo"; state = "Crítico";
    }

    tbody.innerHTML = `
      <tr class="${cls}">
        <td>${diff.toFixed(4)}</td>
        <td>${text}</td>
        <td>${state}</td>
      </tr>
    `;

    // ===============================
    // Imágenes
    // ===============================
    document.getElementById("confusion-img").src =
        `${API_BASE_URL}${m.confusion_matrix_image}`;

    document.getElementById("roc-img").src =
        `${API_BASE_URL}${m.roc_curve_image}`;
}