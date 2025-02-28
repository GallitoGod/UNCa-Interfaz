const body = document.querySelector('body');
const modeIcon = document.getElementById('mode-icon');

export function enableDarkMode() {
  try {
    body.style.backgroundColor = '#232323';
    modeIcon.src = '../../static/images/darkMode.svg';
  } catch (err) {
    console.error('enableDarkMode Error:', err);
  }
}

export function disableDarkMode() {
  try {
    body.style.backgroundColor = '#ffffff';
    modeIcon.src = '../../static/images/lightMode.svg';
  } catch (err) {
    console.error('disableDarkMode Error:', err);
  }
}