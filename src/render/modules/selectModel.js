import { selectModelUrl } from "./constants.js";

export async function selectModel(modelName) {
    if (!modelName) return;
    try {
        const response = await fetch(selectModelUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: modelName }),
        });
        const data = await response.json();
        console.log("Modelo seleccionado:", data);
    } catch (err) {
        console.error("Error al seleccionar modelo:", err);
    }
}