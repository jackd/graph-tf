import contextlib
import os
import shutil
import tempfile


@contextlib.contextmanager
def tempfile_context():
    path = tempfile.mktemp()
    try:
        yield path
    finally:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
