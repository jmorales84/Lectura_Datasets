document.addEventListener("DOMContentLoaded", async () => {
    const status = document.getElementById("status");
    const datasetId = sessionStorage.getItem("dataset_id");

    if (!datasetId) {
        status.textContent = "No hay dataset cargado. Regresa al menú.";
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/apis/prepare_dataset/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_id: datasetId })
        });

        if (!response.ok) throw new Error("Error procesando dataset");

        const data = await response.json();
        status.textContent = "Preparación completada ✔";

        // --- Split info ---
        const splitDiv = document.getElementById("splitInfo");
        splitDiv.innerHTML = `
            <h3>Tamaños de los splits</h3>
            <table class="table">
                <tr><th>Split</th><th>Filas</th></tr>
                <tr><td>Train</td><td>${data.split_sizes.train}</td></tr>
                <tr><td>Validation</td><td>${data.split_sizes.validation}</td></tr>
                <tr><td>Test</td><td>${data.split_sizes.test}</td></tr>
            </table>
        `;

        // --- Imputer info ---
        const imputerDiv = document.getElementById("imputerInfo");
        let imputerRows = "";
        for (const [col, val] of Object.entries(data.imputer_median)) {
            imputerRows += `<tr><td>${col}</td><td>${val}</td></tr>`;
        }
        imputerDiv.innerHTML = `
            <h3>Medianas usadas para imputación</h3>
            <table class="table">
                <tr><th>Columna</th><th>Mediana</th></tr>
                ${imputerRows}
            </table>
        `;

        // --- One-hot columns ---
        const oneHotDiv = document.getElementById("oneHotInfo");
        oneHotDiv.innerHTML = `
            <h3>Columnas one-hot encoding</h3>
            <ul>${data.one_hot_columns.map(c => `<li>${c}</li>`).join("")}</ul>
        `;

        // --- Scaled columns ---
        const scaledDiv = document.getElementById("scaledInfo");
        scaledDiv.innerHTML = `
            <h3>Columnas escaladas</h3>
            <ul>${data.scaled_columns.map(c => `<li>${c}</li>`).join("")}</ul>
        `;

        // --- Vista previa X_train (tabla interactiva) ---
        const previewDiv = document.getElementById("previewTable");
        const previewData = data.X_train_preview;
        if(previewData.length > 0){
            const headers = Object.keys(previewData[0]);
            const headerRow = headers.map(h => `<th>${h}</th>`).join("");
            const bodyRows = previewData.map((row, idx) => {
                return "<tr>" + headers.map(h => `<td>${row[h]}</td>`).join("") + "</tr>";
            }).join("");
            previewDiv.innerHTML = `
                <h3>Vista previa X_train (primeras 10 filas)</h3>
                <table class="table">
                    <thead><tr>${headerRow}</tr></thead>
                    <tbody>${bodyRows}</tbody>
                </table>
            `;
        } else {
            previewDiv.innerHTML = "<p>No hay datos para mostrar</p>";
        }

    } catch (err) {
        console.error(err);
        status.textContent = "Error al procesar el dataset";
    }
});
