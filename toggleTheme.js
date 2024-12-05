const body = document.querySelector('body');
const words = document.querySelector('label');
const modeIcon = document.getElementById('mode-icon');
const settingsMenu = document.getElementById('settings-menu');
const mainMenu = document.getElementById('main-menu');

export function enableDarkMode() {
  body.style.backgroundColor = '#232323';
  modeIcon.src = 'images/darkMode.svg';
  settingsMenu.style.backgroundColor = '#3333333';
  mainMenu.style.backgroundColor = '#3333333';
  words.style.color = '#111';
}

export function disableDarkMode() {
  body.style.backgroundColor = '#ffffff';
  modeIcon.src = 'images/lightMode.svg';
  settingsMenu.style.backgroundColor = '#ffffff';
  mainMenu.style.backgroundColor = '#ffffff';
  words.style.color = '#333333';
}
