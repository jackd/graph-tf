import functools
import typing as tp

import gin

register = functools.partial(gin.register, module="gtf.utils.misc")


@register
def call(fn: tp.Callable):
    return fn()


@register
def maybe_double(x: int, cond: bool):
    if cond:
        return 2 * x
    return x
