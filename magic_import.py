import logging
import os
import re
import sys
import tempfile
from importlib.machinery import PathFinder, ModuleSpec
from importlib._bootstrap_external import _path_split
from importlib.abc import SourceLoader
from pathlib import Path
from shutil import rmtree
import subprocess


logger = logging.getLogger('magic_import')


def is_package(fullname, filename):
    filename_base = filename.rsplit('.', 1)[0]
    tail_name = fullname.rpartition('.')[2]
    return filename_base == '__init__' and tail_name != '__init__'


def get_site_packages_dir():
    try:
        import site
        site_packages = site.getsitepackages()
        site_packages.append(site.getusersitepackages())
    except AttributeError:
        # https://github.com/pypa/virtualenv/issues/737
        from distutils.sysconfig import get_python_lib
        site_packages = [get_python_lib()]
    return site_packages


class MagicPackSourceLoader(SourceLoader):
    def __init__(self, name, module_info, fake_root, path):
        self.name = name
        self.module_info = module_info
        self.fake_root = fake_root
        self.path = path

    def is_package(self, fullname):
        """Concrete implementation of InspectLoader.is_package by checking if
        the path returned by get_filename has a filename of '__init__.py'."""
        filename = _path_split(self.module_info[3])[1]
        return is_package(fullname, filename)

    def get_filename(self, fullname):
        return os.path.join(self.fake_root,
                            self.module_info[3].rsplit("/", 1)[-1])

    def get_data(self, path):
        return self.module_info[1]

    def exec_module(self, module):
        super(MagicPackSourceLoader, self).exec_module(module)
        module.__file__ = self.get_filename(self.name)


def restore_assets(path, assets):
    for asset, (content, stat) in assets.items():
        asset_path = os.path.join(path, asset)
        os.makedirs(os.path.dirname(asset_path), exist_ok=True)
        with open(asset_path, "wb") as f:
            f.write(content)
        os.chmod(asset_path, stat.st_mode)


already_prompt = set()


def find_package_version(package_name):
    package_name = package_name.replace("-", "_")
    site_packages = get_site_packages_dir()
    for path in site_packages:
        if os.path.isdir(path):
            for dist_info_path in os.listdir(path):
                m = re.search(rf"{package_name}-(.*)\.dist-info", str(dist_info_path))
                if m:
                    return m.group(1)
                m = re.search(rf"{package_name}-(.*)-py[\d.]+\.egg-info", str(dist_info_path))
                if m:
                    return m.group(1)
    return "(can't find)"


def extract_cython_files(main_path, fullname, sources, assets):
    pkg_path = os.path.join(*fullname.split(".")[:-1]) if "." in fullname else ""
    path_with_pkg = os.path.join(main_path, pkg_path)
    os.makedirs(path_with_pkg, exist_ok=True)
    path = Path(path_with_pkg)
    while path != Path(main_path):
        with open(str(path) + "/__init__.py", "w") as f:
            pass
        path = path.parent
    pyx_sources, pyxbld_sources, pyxdep_sources = sources
    last_name = fullname.split(".")[-1]
    pyx_path = os.path.join(path_with_pkg, last_name) + ".pyx"
    with open(pyx_path, "w") as f:
        f.write(pyx_sources)
    if pyxbld_sources is not None:
        with open(os.path.join(path_with_pkg, last_name) + ".pyxbld", "w") as f:
            f.write(pyxbld_sources)
    if pyxdep_sources is not None:
        with open(os.path.join(path_with_pkg, last_name) + ".pyxdep", "w") as f:
            f.write(pyxdep_sources)
    restore_assets(path_with_pkg, assets)
    return pyx_path


class VersionPromptProxyLoader(object):
    def __init__(self, name, original_loader, pypi_name, pypi_version):
        self.name = name
        self.original_loader = original_loader
        self.pypi_name = pypi_name
        self.pypi_version = pypi_version

        if hasattr(original_loader, "is_package"):
            self.is_package = original_loader.is_package

        # some loader has no new style api
        if hasattr(original_loader, "create_module"):
            self.create_module = original_loader.create_module
            self.exec_module = self.exec_module_

    @classmethod
    def check_version(cls, pypi_name, pypi_version):
        if pypi_name in already_prompt:
            return

        new_version = find_package_version(pypi_name)
        if new_version != pypi_version:
            print(f"Version of package {pypi_name} changes from "
                  f"{pypi_version} to {new_version}", file=sys.stderr)
        already_prompt.add(pypi_name)

    def exec_module_(self, module):
        ret = self.original_loader.exec_module(module)
        self.check_version(self.pypi_name, self.pypi_version)
        return ret

    def load_module(self, fullname):
        module = self.original_loader.load_module(fullname)
        self.check_version(self.pypi_name, self.pypi_version)
        return module


def get_pyx_loader():
    from pyximport import PyxLoader
    import base64

    class MyPyxLoader(PyxLoader):
        def __init__(self, fullname, sources, assets, init_path=None,
                     inplace=False, language_level=None):
            self.main_path = os.path.join(tempfile.gettempdir(),
                                          base64.b64encode(os.urandom(15)).decode()) + "/"
            # required to load .pxd
            sys.path.append(self.main_path)
            pyx_path = extract_cython_files(self.main_path, fullname, sources, assets)

            super().__init__(fullname, pyx_path, init_path, self.main_path + ".cache/",
                             inplace, language_level)

        def load_module(self, fullname):
            ret = super(MyPyxLoader, self).load_module(fullname)
            rmtree(self.main_path)
            sys.path.remove(self.main_path)
            return ret

    return MyPyxLoader


class MagicPackImporter(PathFinder):
    def __init__(self, envir_info):
        super(MagicPackImporter, self).__init__()
        for name, (func, value) in envir_info["system"].items():
            new_value = func()
            if new_value != value:
                print(f"{name} chages from {value} to {new_value}")
        self.module_sources = envir_info["modules"]

        for name in sys.modules.keys():
            module_info = self.module_sources.get(name)
            if module_info:
                if module_info[0] != "version":
                    print(f"{name} is already loaded, ",
                          "codes in magic pack won't be used. ",
                          file=sys.stderr)
                else:
                    pypi_name, pypi_version = module_info[-2:]
                    VersionPromptProxyLoader.check_version(pypi_name, pypi_version)

        self.auto_install = False
        self.pyx_loader_class = None

    def find_spec(self, fullname, path=None, target=None):
        module_info = self.module_sources.get(fullname)

        if module_info:
            if module_info[0] == "source":
                filename = _path_split(module_info[3])[1]
                assets = module_info[4]
                tmp_dir = tempfile.gettempdir()
                pkg_tmp_dir = os.path.join(tmp_dir, "magic_pack", fullname)
                if assets:
                    restore_assets(pkg_tmp_dir, assets)
                spec = ModuleSpec(
                    name=fullname,
                    loader=MagicPackSourceLoader(fullname, module_info, pkg_tmp_dir, path),
                    is_package=is_package(fullname, filename)
                )
                return spec
            if module_info[0] == "version":
                default_spec = super(MagicPackImporter, self).find_spec(fullname, path, target)
                pypi_name, pypi_version = module_info[-2:]
                # this module is not installed
                if default_spec is None:
                    result = None
                    while not self.auto_install and result not in ("y", "n", "all"):
                        result = input(f"Package {pypi_name}({pypi_version}) is not installed. "
                                       f"Do you want to pip install it now? (y/n/all)")
                    if result == "all":
                        self.auto_install = True
                    if self.auto_install or result == "y":
                        # some error may occur if run pip in current interpreter
                        subprocess.call(
                            [sys.executable, "-m", "pip", "install",
                             f"{pypi_name}=={pypi_version}", "--no-dependencies"]
                        )
                    if result == "n":
                        return None
                    default_spec = super(MagicPackImporter, self).find_spec(fullname, path, target)
                    # sometimes installation failed
                    if default_spec is None:
                        return None
                loader = VersionPromptProxyLoader(
                    fullname, default_spec.loader, *module_info[-2:])
                default_spec.loader = loader
                return default_spec
            if module_info[0] == "cython_source":
                for importer in sys.meta_path:
                    from pyximport import PyxImporter
                    if isinstance(importer, PyxImporter):
                        global pyxargs
                        if fullname in sys.modules and not pyxargs.reload_support:
                            return None  # only here when reload()
                        default_spec = ModuleSpec(name=fullname, loader=None)
                        if self.pyx_loader_class is None:
                            self.pyx_loader_class = get_pyx_loader()
                        # noinspection PyCallingNonCallable
                        loader = self.pyx_loader_class(fullname, module_info[1], module_info[2],
                                                       inplace=importer.inplace,
                                                       language_level=importer.language_level)
                        default_spec.loader = loader
                        return default_spec
        return None

    def install(self):
        sys.meta_path.insert(0, self)

    def uninstall(self):
        sys.meta_path.remove(self)
