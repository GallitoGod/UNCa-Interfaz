import { loadModelUrl } from "./constants.js";

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
  } catch (error) {
    console.error("Error al cargar los modelos:", error);
  }
}
  