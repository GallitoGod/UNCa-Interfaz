# tasks/errors.py — excepciones tipadas de la frontera del seam.
# Se traducen a HTTP en la API (mainAPI._load_and_validate):
#   - UnknownModelType  -> 422 (subclase de ValueError)
#   - TaskNotImplemented -> 501 (subclase de NotImplementedError)
# El subclaseo es deliberado: los handlers existentes de la API ya mapean
# ValueError->422 y NotImplementedError->501, asi que no hay que tocar la frontera.


class UnknownModelType(ValueError):
    """El model_type del JSON no esta registrado en el registry de estrategias."""


class TaskNotImplemented(NotImplementedError):
    """El model_type es reconocido pero su pipeline todavia no esta implementado."""
