export let selectedModel = null;

const port = 8000;
const localHost = `http://127.0.0.1:${ port }`;

export const loadModelUrl = `${ localHost }/get_models`;
export const selectModelUrl = `${ localHost }/select_model`;
export const predictUrl = `${ localHost }/predict`;