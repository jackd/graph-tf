"""
Generic entry-point for programs predominantly configured via `gin-config`.

Parses arbitrary config files and bindings. It is assumed that one of these
configures a `main.fn`.

Positional args are interpreted as paths to config files, with `.gin` appended
if missing. The environment variable `KB_CONFIG` is defined before passing as
`kblocks/configs` (though be sure to include quotes if it is not defined
in your environment). See the files contained there for examples that declare
undefined macros (which are expected to be defined in other config files).

Usage:
```python
python -m kblocks '$KB_CONFIG/fit.gin' rel/path/to/config \\
    --bindings='f.x = 3' --bindings='
        y = @g()
        g.param = 5
    ' \\
    --log_dir=/tmp/my_log  # absl flag
```

General notes:
    - bindings may be repeated or separated by new-lines
    - by default, included files in config files are interpreted first as
        relative to the directory containing the config file, and if missing
        relative to the current working directory. You can disable relative
        includes using `--incl_rel=False`
"""


import gacl
import graph_tf.config  # pylint: disable=unused-import

if __name__ == "__main__":
    gacl.cli_main()
