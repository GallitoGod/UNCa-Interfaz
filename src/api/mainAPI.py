# Hasta empezar a darle sentido al script mainAPI.py, este va ser un bloc de notas para construir y reparar, si es necesario, la logica
# y la arquitectura general del programa. Basando todo en conseguir una estructura modular en un formato strategy y totalmente escalable
# a futuro.

# En base a ello:

# --------------08/06/2025--------------
# Encontre una inconsistencia conceptual en la estructura adaptador-transformador:
#     En varios sentidos el transformador actua dependiente a la IA en uso, en cuento su objetivo
#     es ser un proceso universal y agnostico al modelo. Con relacion a los adaptadores, estos
#     deben estar diseñados para cumplir con los requisitos internos del modelo en uso, sean cuales sean.
#     Para poder confiar en los transformers tendria que cumplir la pregunta de: 
#         ¿Podría YOLOv5 y un clasificador MobileNet usar esto tal cual está?
#---------------------------------------





# 🔁 Revisiones:

#     “limpieza de transformers”
#     1. Auditoria completa de los transformers actuales
#         Verificar cada transformer actual y preguntarse:
#             ¿Esto se puede aplicar a cualquier modelo sin romper su inferencia?

#     2. Detectar y eliminar logica contextual
#     Ejemplos comunes que deberían salir de los transformers:
#         Cambios de canal de color (BGR↔RGB)
#         Normalizaciones con valores especificos (e.g., ImageNet)
#         Cambios de orden de ejes transpose() si no es generico
#         Rescalados a un tamaño fijo que solo usa un modelo

#     3. Reforzar la interfaz entre adapter y transformer
#     Idealmente, los adapters deberian llamar a transformers que se comportan como funciones puras.
#     Y que nunca un transformer modifique el input de una manera que el adapter no controle.

# ⚠️ inconsistencias:

# ⚠️ Tambien es importante recordar la eliminacion de las contribuciones anteriores de "model_loader.py" (funcion de inferencia, 
# y adaptadores por modelo y arquitectura). La arquitectura actual es mas sana,pero es crucial completar este corte de 
# responsabilidades, porque aun hay rastros conceptuales de ese viejo diseño.

# ⚠️ Todavia no se implementa el cambio de confianza desde el cliente en los transformadores. Aunque antes de hacerlo 
# es necesario pasar por la auditoria de "transformers.py".


# 💡 Ideas:

# 💡 Falta logging del ciclo de vida de la IA activa
# Seria una buena idea agregar un sistema de log por modelo activo. Aunque sea en consola o un archivo .log,
# ayuda muchísimo en debugging y al implementar adaptadores.