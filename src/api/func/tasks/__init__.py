# tasks/ — seam por model_type del backend.
# Cada tipo de modelo (detection / classification / segmentation) tiene una
# TaskStrategy que posee: como armar su pipeline, que resultado de dominio produce
# y como se serializa al cliente. El registry despacha por model_type.
