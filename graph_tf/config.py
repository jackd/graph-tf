import os

import gin

folder = os.path.dirname(__file__)
for path in (folder, os.path.join(folder, "projects")):
    gin.config.add_config_file_search_path(path)
del folder
