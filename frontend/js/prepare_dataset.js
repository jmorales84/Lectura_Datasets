/**
 * Lógica para la etapa de Preparación y Transformación del Dataset
 */
document.addEventListener("DOMContentLoaded", async () => {
    const status = document.getElementById("status");
    const datasetId = sessionStorage.getItem("dataset_id");

    // Verificación inicial de seguridad
    if (!datasetId || datasetId === "null") {
        status.innerHTML = '<b class="text-danger">⚠️ No se encontró un Dataset activo. Regresa a "Subir Archivo".</b>';
        return;
    }

    try {
        status.textContent = "⏳ Procesando transformaciones en el servidor...";

        const response = await fetch(`${API_BASE_URL}/apis/prepare_dataset/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_id: datasetId })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Error interno en el servidor");
        }

        const data = await response.json();
        status.innerHTML = '<span class="text-success">✅ Preparación completada con éxito</span>';

        // 1. Mostrar información de los Splits
        renderSplitCards(data.split_sizes);

        // 2. Mostrar medianas calculadas (SimpleImputer)
        renderMediansTable(data.imputer_median);

        // 3. Mostrar columnas transformadas
        renderTransformationsInfo(data.one_hot_columns, data.scaled_columns);

        // 4. Mostrar Vista Previa de X_train transformado
        renderPreviewTable(data.X_train_preview);

    } catch (err) {
        console.error("Error:", err);
        status.innerHTML = `<b class="text-danger">❌ Error: ${err.message}</b>`;
    }
});

/** RENDERIZADORES **/

function renderSplitCards(sizes) {
    const container = document.getElementById("splitInfo");
    container.innerHTML = `
        <div class="row text-center mt-3">
            <div class="col-md-4">
                <div class="card bg-light p-3 border-success">
                    <h6>Train (60%)</h6>
                    <h3 class="text-success">${sizes.train}</h3>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-light p-3 border-warning">
                    <h6>Validation (20%)</h6>
                    <h3 class="text-warning">${sizes.validation}</h3>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-light p-3 border-primary">
                    <h6>Test (20%)</h6>
                    <h3 class="text-primary">${sizes.test}</h3>
                </div>
            </div>
        </div>`;
}

function renderMediansTable(medians) {
    const container = document.getElementById("imputerInfo");
    const entries = Object.entries(medians);
    
    if (entries.length === 0) {
        container.innerHTML = "<p class='text-muted'>No se aplicó imputación de nulos.</p>";
        return;
    }

    let rows = entries.map(([col, val]) => `
        <tr>
            <td><code>${col}</code></td>
            <td class="text-end font-monospace">${val.toFixed(4)}</td>
        </tr>`).join("");

    container.innerHTML = `
        <div class="table-responsive mt-3">
            <table class="table table-sm table-bordered">
                <thead class="table-secondary"><tr><th>Atributo Numérico</th><th class="text-end">Valor Mediana</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

function renderTransformationsInfo(oneHot, scaled) {
    document.getElementById("oneHotInfo").innerHTML = `
        <div class="alert alert-info py-2">
            <small><b>One-Hot Encoding:</b> Se generaron ${oneHot.length} nuevas columnas binarias.</small>
        </div>`;
    
    document.getElementById("scaledInfo").innerHTML = `
        <div class="alert alert-secondary py-2">
            <small><b>Robust Scaling:</b> Atributos normalizados: ${scaled.join(", ")}</small>
        </div>`;
}

function renderPreviewTable(rows) {
    const container = document.getElementById("previewTable");
    if (!rows || rows.length === 0) return;

    const headers = Object.keys(rows[0]);
    
    let headerHtml = headers.map(h => `<th>${h}</th>`).join("");
    let rowsHtml = rows.map(row => `
        <tr>${headers.map(h => `<td>${row[h] !== null ? row[h] : '<em>NaN</em>'}</td>`).join("")}</tr>
    `).join("");

    container.innerHTML = `
        <h5 class="mt-4">Vista Previa de Datos Transformados (X_train)</h5>
        <div class="table-responsive shadow-sm">
            <table class="table table-hover table-bordered table-sm custom-table">
                <thead class="table-dark"><tr>${headerHtml}</tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>`;
}