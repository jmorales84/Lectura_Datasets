function renderMeta(data) {
    document.getElementById("dataset-meta").innerHTML = `
        <p><strong>Archivo:</strong> ${data.filename}</p>
        <p><strong>Filas:</strong> ${data.rows}</p>
        <p><strong>Columnas:</strong> ${data.columns}</p>
    `;
}

function renderPreview(preview) {
    let html = "<table class='table'><thead><tr>";
    preview.columns.forEach(c => html += `<th>${c}</th>`);
    html += "</tr></thead><tbody>";

    preview.rows.forEach(row => {
        html += "<tr>";
        preview.columns.forEach(c => html += `<td>${row[c]}</td>`);
        html += "</tr>";
    });

    html += "</tbody></table>";
    document.getElementById("preview-table").innerHTML = html;
}

function renderInfo(info) {
    let html = `<table class="table"><thead>
        <tr><th>Columna</th><th>Tipo</th><th>No Nulos</th></tr>
    </thead><tbody>`;

    info.columns.forEach(c => {
        html += `<tr>
            <td>${c.name}</td>
            <td>${c.dtype}</td>
            <td>${c.non_null}</td>
        </tr>`;
    });

    html += "</tbody></table>";
    document.getElementById("dataset-info").innerHTML = html;
}

function renderPlots(plots) {
    const container = document.getElementById("plots-container");
    container.innerHTML = "";

    plots.forEach(p => {
        const imgSrc = p.image_base64 ? p.image_base64 : `${API_BASE_URL}${p.url}`;
        container.innerHTML += `
            <div class="plot-card">
                <h4>${p.column || p.title}</h4>
                <img src="${imgSrc}">
            </div>`;
    });
}

function renderCorrelationPlots(plots) {
    const container = document.getElementById("plots-container");

    plots.forEach(p => {
        const imgSrc = p.image_base64 ? p.image_base64 : `${API_BASE_URL}${p.url}`;
        container.innerHTML += `
            <div class="plot-card">
                <h4>${p.title}</h4>
                <img src="${imgSrc}">
            </div>`;
    });
}

function renderConfusionMatrix(plots) {
    const container = document.getElementById("plots-container");

    plots.forEach(p => {
        const imgSrc = p.image_base64 ? p.image_base64 : `${API_BASE_URL}${p.url}`;
        container.innerHTML += `
            <div class="plot-card">
                <h4>${p.title}</h4>
                <img src="${imgSrc}">
            </div>`;
    });
}

document.addEventListener("DOMContentLoaded", async () => {
    const datasetId = sessionStorage.getItem("dataset_id");
    if (!datasetId) return alert("Dataset no cargado");

    const res = await fetch(`${API_BASE_URL}/apis/visualizar_dataset/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId })
    });

    const data = await res.json();

    renderMeta(data);
    renderPreview(data.preview);
    renderInfo(data.info);
    renderPlots(data.plots);
    renderCorrelationPlots(data.correlation_plots);
    renderConfusionMatrix(data.confusion_matrix);
});

document.getElementById("btn-back").onclick = () => {
    window.location.href = "index.html";
};
