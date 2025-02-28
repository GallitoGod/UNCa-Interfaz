import { selectedModel, selectModelUrl } from "./constants.js";

export async function selectModel() {

    try {
        const response = await fetch(selectModelUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model_name: selectedModel,
            }),
        });
        console.log("server response:", response);
    } catch (err) {
        console.error("Error sending POST request:", err);
    }
}