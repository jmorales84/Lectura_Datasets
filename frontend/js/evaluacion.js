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

        const startData = await startResponse.json();
        const taskId = startData.task_id;

        /* ===============================
           2. POLLING DE ESTADO
        =============================== */
        let finished = false;

        while (!finished) {
            const estadoRes = await fetch(
                `${API_BASE_URL}/apis/estado_modelo/${taskId}/`
            );

            if (!estadoRes.ok) {
                throw new Error("Error consultando el estado del modelo");
            }

            const estado = await estadoRes.json();

            // Actualizar UI de progreso
            const progress = estado.progress ?? 0;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = estado.message || "Procesando...";

            if (estado.status === "done") {
                finished = true;
                progressBar.style.width = "100%";
                progressText.textContent = "Evaluación completada";
                renderResultados(estado.result);
            }

            if (estado.status === "error") {
                throw new Error(estado.error);
            }

            // Esperar antes del siguiente poll
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
    const accVal = data.metrics.accuracy_validation;
    const accTest = data.metrics.accuracy_test;
    const diff = data.metrics.accuracy_difference;

    document.getElementById("acc-val").textContent = accVal.toFixed(4);
    document.getElementById("acc-test").textContent = accTest.toFixed(4);
    document.getElementById("acc-diff").textContent = diff.toFixed(4);

    const tbody = document.querySelector("#interpretation-table tbody");

    let cls, text, state;

    if (diff < 0.02) {
        cls = "ok";
        text = "Excelente generalización";
        state = "Óptimo";
    } else if (diff < 0.05) {
        cls = "ok";
        text = "Generalización normal";
        state = "Aceptable";
    } else if (diff < 0.10) {
        cls = "warn";
        text = "Posible overfitting";
        state = "Riesgo";
    } else {
        cls = "bad";
        text = "Overfitting severo";
        state = "Crítico";
    }

    tbody.innerHTML = `
        <tr class="${cls}">
            <td>${diff.toFixed(4)}</td>
            <td>${text}</td>
            <td>${state}</td>
        </tr>
    `;

    document.getElementById("confusion-img").src =
        `${API_BASE_URL}${data.metrics.confusion_matrix_image}`;
}
