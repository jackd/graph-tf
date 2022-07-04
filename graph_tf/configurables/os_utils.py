import functools
import os
import typing as tp

import gin

register = functools.partial(gin.register, module="gtf.os_utils")


@register
def expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


@register
def join(a, p):
    return os.path.join(a, p)


@register
def join_star(args):
    return os.path.join(*args)


@register
def get_environ(key: str, default: tp.Optional[tp.Any] = None):
    value = os.environ.get(key)
    if value is None:
        return default
    return value


@register
def with_extension(prefix: str, extension: str) -> str:
    if extension.startswith("."):
        return f"{prefix}{extension}"
    return f"{prefix}.{extension}"
