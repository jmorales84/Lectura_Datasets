document.addEventListener("DOMContentLoaded", async () => {
    const status = document.getElementById("status");
    const results = document.getElementById("results");
    const datasetId = sessionStorage.getItem("dataset_id");

    if (!datasetId) {
        status.textContent = "No hay dataset cargado.";
        return;
    }

    try {
        //  Llamada al endpoint comprimido
        const response = await fetch(`${API_BASE_URL}/apis/pipelines_personalizados/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_id: datasetId })
        });

        //  Leer como texto (NO response.json())
        const compressedText = await response.text();

        //  Parsear JSON de la envoltura {"compressed": "..."}
        const wrapper = JSON.parse(compressedText);
        if (!wrapper.compressed) throw new Error("No hay datos comprimidos en la respuesta.");

        //  Decodificar base64
        const strData = atob(wrapper.compressed);

        //  Convertir a Uint8Array para pako
        const uint8 = new Uint8Array(strData.split("").map(c => c.charCodeAt(0)));

        //  Descomprimir con pako
        const decompressed = pako.ungzip(uint8, { to: "string" });

        //  Parsear JSON final
        const data = JSON.parse(decompressed);

        status.textContent = "Procesamiento completado";

        //  Renderizar tablas
        renderSection("X_train original", data.x_train_original);
        renderSection("X_train con NaN", data.x_train_with_nan);
        renderSection("Después de DeleteNanRows", data.after_delete_nan);
        renderSection("Después de CustomScaler", data.after_scaler);
        renderSection("Después de OneHotEncoding", data.after_onehot);
        renderSection("Resultado final Pipeline", data.pipeline_final);

    } catch (error) {
        status.textContent = "Error al procesar el dataset";
        console.error(error);
    }

    function renderSection(title, tableData) {
        if (!tableData || tableData.length === 0) return;

        const section = document.createElement("div");
        section.className = "table-container";

        section.innerHTML = `
            <h2 class="section-title">
                <i class="fa-solid fa-table"></i> ${title}
            </h2>
            ${createTable(tableData)}
        `;

        results.appendChild(section);
    }

    function createTable(data) {
        const headers = Object.keys(data[0]);
        let html = `<table class="data-table"><thead><tr>`;

        headers.forEach(h => html += `<th>${h}</th>`);
        html += `</tr></thead><tbody>`;

        //  Limitar a 10 filas
        data.slice(0, 10).forEach(row => {
            html += `<tr>`;
            headers.forEach(h => html += `<td>${row[h]}</td>`);
            html += `</tr>`;
        });

        html += `</tbody></table>`;
        return html;
    }
});
