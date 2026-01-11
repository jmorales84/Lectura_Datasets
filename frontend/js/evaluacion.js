document.addEventListener("DOMContentLoaded", async () => {

    const datasetId = sessionStorage.getItem("dataset_id");
    if (!datasetId) {
        alert("No hay dataset cargado");
        return;
    }

    const response = await fetch(`${API_BASE_URL}/apis/evaluar_modelo/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId })
    });

    const data = await response.json();

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
});
