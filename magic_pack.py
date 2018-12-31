"""
Pack python modules into single executeable file
"""
import base64
import os
import platform
import struct
import sys
import inspect
from pathlib import Path
from coli.parser_tools import magic_load, magic_import, interpreter_selector
from imp import is_builtin

from coli.parser_tools.magic_load import read_script, read_until_entrance

script_template = b'''#!/bin/sh
# MAGIC_STRING = SUGAR_RUSH
SCRIPT_SIZE={scriptsize}
SCRIPT_FILE=$(mktemp)

# Script 1: get python path
# (getopt not support long options on some system
#  so we use python)
PYTHON=$(which python3)
cat << EOF > $SCRIPT_FILE
# ===== Script1 START =====
{interpreter_selector_content}
# ====== Script1 END ======
EOF


# first time: specify interpreter if needed
PYTHON=$($PYTHON "$SCRIPT_FILE" "$@")
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    exit $RETVAL
fi

# second time: specify virtualenv if needed
PYTHON=$(USE_VENV=true $PYTHON "$SCRIPT_FILE" "$@")
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    exit $RETVAL
fi

cat << EOF > $SCRIPT_FILE
# ===== Script2 START =====
{magic_load_content}
# ====== Script2 END ======
EOF

exec $PYTHON "$SCRIPT_FILE" "$@"

# some error may cause the script run here
exit 1


___BINARY_START___
'''


def read_file(path):
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    return None


def get_cython_files(fullname, package_path):
    from pyximport import PYX_EXT, PYXBLD_EXT, PYXDEP_EXT
    mod_parts = fullname.split('.')
    module_name = mod_parts[-1]
    cython_files = [module_name + i for i in (PYX_EXT, PYXBLD_EXT, PYXDEP_EXT)]
    paths = package_path or sys.path
    for path in paths:
        if not path:
            path = os.getcwd()
        elif not os.path.isabs(path):
            path = os.path.abspath(path)

        cython_paths = [os.path.join(path, i) for i in cython_files]

        if os.path.isfile(cython_paths[0]):
            contents = tuple(read_file(i) for i in cython_paths)
            # if pyxbld exists, get assets
            if contents[1] is not None:
                namespace = {}
                exec(contents[1], namespace)
                assets = namespace.get("__assets__") or []
            else:
                assets = []
            asset_sources = collect_assets(path, assets)
            return contents, asset_sources
    return (None, None, None), {}


def collect_assets(path, assets):
    asset_sources = {}
    for asset_item in assets:
        path_obj = Path(path)
        for asset in path_obj.glob(asset_item):
            if not str(asset).startswith(str(path_obj)):
                print(f"asset {asset} not in {path_obj}, ignore it.")
                continue
            # noinspection PyTypeChecker
            with open(asset, "rb") as f:
                asset_sources[asset.relative_to(path_obj)] = (f.read(), asset.stat())
    return asset_sources


def write_script(writer):
    script_no_size = script_template.replace(
        b"{magic_load_content}",
        inspect.getsource(magic_load).encode("utf-8")).replace(
        b"{interpreter_selector_content}",
        inspect.getsource(interpreter_selector).encode("utf-8"))
    script_size = base64.b64encode(struct.pack("!Q", len(script_no_size)))
    # len("{scriptsize}") == 12 == len(script_size), the size won't change
    script = script_no_size.replace(b"{scriptsize}", script_size)
    writer.write(script)


def get_name_version(file_name):
    metadata = {}
    with open(file_name, "r") as f:
        for line in f:
            fields = line.split(": ", 1)
            if len(fields) == 2:
                key, value = fields
                metadata[key] = value.strip()
    name = metadata.get("Name")
    version = metadata.get("Version")
    return name, version


def get_dist_info():
    site_packages = magic_import.get_site_packages_dir()
    results = {}
    for path in site_packages:
        path = Path(path)
        for dist_info_path in path.glob("*.dist-info"):
            try:
                name, version = get_name_version(dist_info_path / "METADATA")
                if not name or not version:
                    continue
                # noinspection PyTypeChecker
                with open(dist_info_path / "RECORD", "r") as f:
                    for line in f:
                        file_name, hash, size = line.split(",")
                        if hash:
                            abs_path = path / file_name
                            results[str(abs_path)] = (name, version)
            except (IOError, ValueError):
                continue
        for egg_info_path in path.glob("*.egg-info"):
            try:
                name, version = get_name_version(egg_info_path / "PKG-INFO")
                if not name or not version:
                    continue
                # noinspection PyTypeChecker
                with open(egg_info_path / "installed-files.txt", "r") as f:
                    for line in f:
                        file_name = line.strip()
                        if "__pycache__" not in file_name:
                            abs_path = (egg_info_path / file_name).resolve()
                            results[str(abs_path)] = (name, version)
            except (IOError, ValueError):
                continue
        for egg_path in path.glob("*.egg"):
            if egg_path.is_dir():
                try:
                    name, version = get_name_version(egg_path / "EGG-INFO" / "PKG-INFO")
                    if not name or not version:
                        continue
                    # noinspection PyTypeChecker
                    with open(egg_path / "EGG-INFO" / "SOURCES.txt", "r") as f:
                        for line in f:
                            file_name = line.strip()
                            if "__pycache__" not in file_name:
                                abs_path = (egg_path / file_name).resolve()
                                results[str(abs_path)] = (name, version)
                except (IOError, ValueError) as e:
                    continue
    return results


def get_codes():
    dist_info = get_dist_info()
    base64_file = Path(base64.__file__)
    stdlib_path = base64_file.parent
    # may be link in virtualenv
    if base64_file.is_symlink():
        stdlib_path_2 = Path(os.readlink(base64_file)).resolve().parent
    else:
        stdlib_path_2 = None

    stdlib_paths = [str(i) for i in [stdlib_path, stdlib_path_2] if i is not None]
    module_sources = {}
    for name, module in list(sys.modules.items()):
        if name == "__main__" or name == "__mp_main__":
            continue

        if is_builtin(name):
            continue

        module_file = getattr(module, "__file__", None)
        if not module_file or any(module_file.startswith(i) for i in stdlib_paths):
            continue

        # ignore intellij pydev helper
        if "/helpers/pydev/" in module_file:
            continue

        dist_info_i = dist_info.get(module_file)
        if dist_info_i:
            pypi_name, pypi_version = dist_info_i
            module_sources[name] = ("version", pypi_name, pypi_version)
        elif os.path.isfile(module_file):
            if module_file.endswith(".py"):
                with open(module_file, "rb") as f:
                    source = f.read()
                assets = getattr(module, "__assets__", None)
                if assets is not None:
                    asset_sources = collect_assets(os.path.dirname(module_file), assets)
                else:
                    asset_sources = {}
                module_sources[name] = ("source", source, module.__package__, module.__file__, asset_sources)
            elif module_file.endswith(".so"):
                # may be cython source
                contents, asset_sources = get_cython_files(name, sys.modules[module.__package__].__path__)
                if contents[0] is None:
                    print(f"Cannot find source code of module {name}. "
                          "Storing its binary file...")
                    with open(module_file, "rb") as f:
                        module_sources[name] = ("binary_source", f.read(), module.__package__, module.__file__)
                else:
                    print(f"Storing cython source code of module {name}...")
                    module_sources[name] = ("cython_source", contents, asset_sources)

    envir_info = {"modules": module_sources,
                  "system": {
                      "Python version": (platform.python_version, platform.python_version()),
                      "System architecture": (platform.architecture, platform.architecture()),
                      "Operate system": (platform.system, platform.system())
                  }}
    return inspect.getsource(magic_import), envir_info


def load_parser_from_magic_pack(model_file, new_options=None):
    with open(model_file, "rb") as f:
        read_script(f)
        entrance_class = read_until_entrance(f)
    return entrance_class.load(model_file, new_options)
