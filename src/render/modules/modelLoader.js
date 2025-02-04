export async function fetchModels() {
    const apiEndpoint = "http://127.0.0.1:8000/models";
  
    try {
      const response = await fetch(apiEndpoint);
      const { models } = await response.json();
  
      const selectElement = document.getElementById("modelSelect");
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
  