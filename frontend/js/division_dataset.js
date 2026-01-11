document.addEventListener("DOMContentLoaded", async () => {
    const status = document.getElementById("status");
    const datasetId = sessionStorage.getItem("dataset_id");

    if (!datasetId) {
        status.textContent = "No hay dataset cargado. Regresa al menú.";
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/apis/process_dataset/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: datasetId
            })
        });

        if (!response.ok) {
            throw new Error("Error procesando dataset");
        }

        const data = await response.json();

        status.textContent = "División completada correctamente ✔";

        // Resultados numéricos
        const splitInfo = document.getElementById("splitInfo");
        splitInfo.innerHTML = `
            <h3>Resultados de la división</h3>
            <ul>
                <li>Train: ${data.split_sizes.train}</li>
                <li>Validación: ${data.split_sizes.validation}</li>
                <li>Test: ${data.split_sizes.test}</li>
            </ul>
        `;
        splitInfo.classList.remove("hidden");

        // Gráficas (base64)
        document.getElementById("trainChart").src =
            "data:image/png;base64," + data.histograms.train;
        document.getElementById("valChart").src =
            "data:image/png;base64," + data.histograms.validation;
        document.getElementById("testChart").src =
            "data:image/png;base64," + data.histograms.test;

    } catch (err) {
        console.error(err);
        status.textContent = "Error al procesar el dataset";
    }
});
