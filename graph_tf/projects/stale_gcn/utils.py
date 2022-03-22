import functools
import os
import tempfile

import gin
import h5py

register = functools.partial(gin.register, module="gtf.stale_gcn.utils")


@register
def get_temp_cache():
    return h5py.File(os.path.join(tempfile.mkdtemp(), "cache.h5"), "a")
