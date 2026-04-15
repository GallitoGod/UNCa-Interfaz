import { loadModelUrl } from "./constants.js";
import { selectModel } from "./selectModel.js";

export async function getModels() {
  try {
    const response = await fetch(loadModelUrl);
    const { models } = await response.json();
    const selectElement = document.getElementById("ia-model");
    selectElement.innerHTML = "";
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      selectElement.appendChild(option);
    });
    // Auto-seleccionar el primer modelo para que el backend lo cargue al arrancar
    if (models.length > 0) {
      await selectModel(models[0]);
    }
  } catch (error) {
    console.error("Error al cargar modelos:", error);
  }
}
