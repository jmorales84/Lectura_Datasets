window.addEventListener("DOMContentLoaded", () => {

    // ================================
    // ELEMENTOS DEL DOM
    // ================================
    const input = document.getElementById("DataSetInput");
    const status = document.getElementById("uploadStatus");
    const indicator = document.getElementById("datasetIndicator");
    const resetBtn = document.getElementById("resetDataset");

    // ================================
    // ACTUALIZAR UI
    // ================================
    function actualizarEstadoUI() {
        const datasetId = sessionStorage.getItem("dataset_id");
        console.log("Dataset en sessionStorage:", datasetId);

        if (datasetId) {
            status.textContent = "Dataset cargado y listo para procesar";
            indicator?.classList.remove("hidden");
            resetBtn?.classList.remove("hidden");

            document.querySelectorAll(".menu-btn").forEach(btn => {
                btn.classList.remove("disabled");
            });

        } else {
            status.textContent = "Ningún archivo cargado";
            indicator?.classList.add("hidden");
            resetBtn?.classList.add("hidden");

            document.querySelectorAll(".menu-btn").forEach(btn => {
                btn.classList.add("disabled");
            });
        }
    }

    // ================================
    // SUBIR DATASET
    // ================================
    input?.addEventListener("change", async () => {
        const file = input.files[0];
        if (!file) return;

        status.textContent = "Subiendo archivo al servidor...";
        indicator?.classList.add("hidden");
        resetBtn?.classList.add("hidden");

        try {
            const formData = new FormData();
            formData.append("dataset", file);

            const response = await fetch(`${API_BASE_URL}/apis/upload_dataset/`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Error al subir dataset");

            const data = await response.json();

            sessionStorage.setItem("dataset_id", data.dataset_id);
            actualizarEstadoUI();

        } catch (err) {
            console.error(err);
            sessionStorage.removeItem("dataset_id");
            status.textContent = "Error al subir el archivo";
            actualizarEstadoUI();
        }

        input.value = "";
    });

    // ================================
    // RESET DATASET
    // ================================
    resetBtn?.addEventListener("click", () => {
        sessionStorage.removeItem("dataset_id");
        status.textContent = "Dataset eliminado. Puedes subir otro.";
        actualizarEstadoUI();
    });

    // ================================
    // PROTECCIÓN DE BOTONES
    // ================================
    document.querySelectorAll('[data-requiere-dataset]').forEach(btn => {
        btn.addEventListener("click", e => {
            const datasetId = sessionStorage.getItem("dataset_id");

            if (!datasetId) {
                e.preventDefault();
                alert("Primero debes cargar un dataset.");
            }
        });
    });

    // ================================
    // ESTADO INICIAL
    // ================================
    actualizarEstadoUI();
});
