const body = document.querySelector('body');
const modeIcon = document.getElementById('mode-icon');

export function enableDarkMode() {
  body.style.backgroundColor = '#232323';
  modeIcon.src = '../../static/images/darkMode.svg';
}

export function disableDarkMode() {
  body.style.backgroundColor = '#ffffff';
  modeIcon.src = '../../static/images/lightMode.svg';
}


export function sectionIA(models, modelSelect) {
  modelSelect.innerHTML = '';
    models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    }) 
}