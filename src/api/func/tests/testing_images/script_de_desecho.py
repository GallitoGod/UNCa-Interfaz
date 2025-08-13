import cv2
from pathlib import Path

width = 1920
height = 1080
img_path = Path(__file__).parent / 'horses_copy.jpg' 
img = cv2.imread(str(img_path))  
img_resized = cv2.resize(img, (width, height))
cv2.imwrite('imagen_redimensionada.jpg', img_resized)
