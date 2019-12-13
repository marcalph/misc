import time
import logging
import functools

# classmethod 
# transforme la méthode en méthode de classe - pas besoin d’instance pour éxécuter la méthode, le premier paramètre est la classe elle même
# pour le code commun à toutes les instances et celles des classes enfants.

# staticmethod
# transforme la méthode en méthode statique - pas besoin d’instance pour éxécuter la méthode, aucun paramètre n’est passé automatiquement à la méthode
# pour le code de type “outil”, mais qui n’es pas particulièrement lié à la classe, pour des raisons d’encapsulation.

# property
# transforme la méthode en propriété - la méthode est déguisée pour ressembler à un attribut, mais l’accès à cet attribut (avec le signe “=”) éxécute le code de la méthode
# pour simplifier les APIs.


def logthis(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(fn.__module__)
        ch = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s %(message)s", "%Y-%m-%d %H:%M")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info(f"start run of function {fn.__name__}")
        t = time.time()
        function = fn(*args, **kwargs)
        logger.info(f"{fn.__name__} done in {time.time()-t:.4f} seconds")
        return function
    return wrapper


@logthis
def decoratorf(number):
    print(number%2 == 0)
    return(number%2 == 0)


@logthis
def decoratorf2(number):
    print(number%2 == 0)
    return(number%2 == 0)


if __name__ == "__main__":
    decoratorf(7)
    decoratorf2(7)
