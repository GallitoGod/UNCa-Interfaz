export async function getModels() {
  try {
    const response = await fetch('http://127.0.0.1:8000/models');
    if (!response.ok) throw new Error(`Error al obtener los modelos: ${response.statusText}`);
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
  