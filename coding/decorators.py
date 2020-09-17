import time
import logging
import functools

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s %(message)s", "%Y-%m-%d %H:%M")
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# classmethod 
# transforme la méthode en méthode de classe - pas besoin d’instance pour éxécuter la méthode, le premier paramètre est la classe elle même
# pour le code commun à toutes les instances et celles des classes enfants.

# staticmethod
# transforme la méthode en méthode statique - pas besoin d’instance pour éxécuter la méthode, aucun paramètre n’est passé automatiquement à la méthode
# pour le code de type “outil”, mais qui n’es pas particulièrement lié à la classe, pour des raisons d’encapsulation.

# property
# transforme la méthode en propriété - la méthode est déguisée pour ressembler à un attribut, mais l’accès à cet attribut (avec le signe “=”) éxécute le code de la méthode
# pour simplifier les APIs.


class CustomFormatter(logging.Formatter):
    """Custom formatter, overrides funcName with value of name_override if it exists"""
    def format(self, record):
        if hasattr(record, 'name_override'):
            record.name = record.name_override
        return super(CustomFormatter, self).format(record)


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = CustomFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)


def logthis(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger.info(f"started", extra={'name_override': fn.__name__})
        t = time.time()
        function = fn(*args, **kwargs)
        logger.info(f"ended ran in {time.time()-t:.3f} seconds", extra={'name_override': fn.__name__})
        return function
    return wrapper


@logthis
def decoratorf(number):
    print(number % 2 == 0)
    return(number % 2 == 0)


@logthis
def test(string):
    print(string)

@logthis
def test2(string):
    print(string)


if __name__ == "__main__":
    decoratorf(7)
    test("string")
    test2("hellow")
    