#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" log utilities
"""

# classmethod
# transforme la méthode en méthode de classe - pas besoin d’instance pour éxécuter la méthode, le premier paramètre est la classe elle même
# pour le code commun à toutes les instances et celles des classes enfants.

# staticmethod
# transforme la méthode en méthode statique - pas besoin d’instance pour éxécuter la méthode, aucun paramètre n’est passé automatiquement à la méthode
# pour le code de type “outil”, mais qui n’es pas particulièrement lié à la classe, pour des raisons d’encapsulation.

# property
# transforme la méthode en propriété - la méthode est déguisée pour ressembler à un attribut, mais l’accès à cet attribut (avec le signe “=”) éxécute le code de la méthode
# pour simplifier les APIs.

import functools
import logging
import os
import time
from inspect import getframeinfo, stack


class CustomFormatter(logging.Formatter):
    """Custom formatter, overrides funcname and filename if provided"""

    def format(self, record):
        if hasattr(record, "funcname_override"):
            record.funcname = record.funcname_override
        return super(CustomFormatter, self).format(record)


def logthis(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        py_file_caller = getframeinfo(stack()[1][0])
        extra_args = {
            "funcname_override": os.path.basename(py_file_caller.filename)
            + "/"
            + fn.__name__
        }
        logger.info("started", extra=extra_args)
        t = time.time()
        function = fn(*args, **kwargs)
        logger.info(f"ended in {time.time()-t:.1f} sec", extra=extra_args)
        return function

    return wrapper


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = CustomFormatter(
    "%(asctime)s - %(levelname)s - %(funcname)s - %(message)s", "%Y-%m-%d"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
