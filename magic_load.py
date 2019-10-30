#!/usr/bin/env python3
"""
This file will be inserted into a shell script,
so dont use single comma.
"""
import os
import pickletools
import sys
import pickle
import struct
import base64
from argparse import ArgumentParser
from contextlib import contextmanager
from types import ModuleType
from typing import Any

magic_header = b"#!/bin/sh\n# MAGIC_STRING = SUGAR_RUSH\nSCRIPT_SIZE="
script_size_len = 12


def read_script(reader):
    """
    part 1: header
    part 2: module_loader, module_sources (pickle)
    part 3: entrance_class (pickle)
    rest
    """
    header = reader.read(len(magic_header))
    if header != magic_header:
        raise TypeError("Not a magic pack!")
    script_size_packed = reader.read(script_size_len)
    script_size = struct.unpack("!Q", base64.b64decode(script_size_packed))[0]
    rest = reader.read(script_size - len(magic_header) - script_size_len)
    return header + script_size_packed + rest


def read_importer_and_source(reader):
    magic_importer_source, module_sources = pickle.load(reader)
    importer_namespace: Any = ModuleType("magic_import")
    exec(magic_importer_source, importer_namespace.__dict__)
    return importer_namespace, module_sources


def read_entrance(reader, importer_namespace, module_sources):
    importer = importer_namespace.MagicPackImporter(module_sources)
    importer.install()
    entrance_class = pickle.load(reader)
    return entrance_class


def read_until_entrance(reader):
    importer_namespace, module_sources = read_importer_and_source(reader)
    if os.environ.get("MAGIC_DEBUGGING"):
        import coli.parser_tools.magic_import as importer_namespace
    return read_entrance(reader, importer_namespace, module_sources)


def extract_codes(all_sources, dest, importer_namespace=None):
    if importer_namespace is None:
        import coli.parser_tools.magic_import as magic_import
        importer_namespace = magic_import

    if not os.path.exists(dest):
        os.makedirs(dest)
    else:
        if os.listdir(dest):
            print(f"{dest} is not empty",
                  file=sys.stderr)
            sys.exit(1)

    pypi_pkgs = set()
    with open(os.path.join(dest, "requirements.txt"), "w") as f_req:
        for name, module_info in all_sources.items():
            if "." in name:
                package_name, module_name = name.rsplit(".", 1)
            else:
                package_name, module_name = "", name
            package_path = os.path.join(dest, *package_name.split("."))
            if module_info[0] == "source":
                source, original_package, original_file, assets = module_info[1:]
                filename = importer_namespace._path_split(original_file)[1]
                if filename == "__init__.py":
                    path = os.path.join(package_path, module_name)
                else:
                    path = package_path
                full_path = os.path.join(path, filename)
                os.makedirs(path, exist_ok=True)
                with open(full_path, "wb") as f:
                    f.write(source)
                importer_namespace.restore_assets(path, assets)
            elif module_info[0] == "cython_source":
                sources, assets = module_info[1:]
                importer_namespace.extract_cython_files(
                    dest, name, sources, assets)
            elif module_info[0] == "version":
                pypi_name, pypi_version = module_info[1:]
                if pypi_name not in pypi_pkgs:
                    f_req.write(pypi_name + "==" + pypi_version + "\n")
                    pypi_pkgs.add(pypi_name)


@contextmanager
def open_magic_pack(model_file, use_old_importer=True):
    f = open(model_file, "rb")
    read_script(f)
    magic_importer_source, module_sources = pickle.load(f)
    if use_old_importer:
        importer_namespace = {}
        exec(magic_importer_source, importer_namespace)
        importer_class = importer_namespace["MagicPackImporter"]
    else:
        from .magic_import import MagicPackImporter
        importer_class = MagicPackImporter
    importer = importer_class(module_sources)
    importer.install()
    entrance_class = pickle.load(f)
    try:
        yield entrance_class
    finally:
        importer.disable()
        f.close()


def remove_option(parser, arg):
    for action in parser._actions:
        if (vars(action)["option_strings"]
            and vars(action)["option_strings"][0] == arg) \
                or vars(action)["dest"] == arg:
            parser._remove_action(action)

    for action in parser._action_groups:
        vars_action = vars(action)
        var_group_actions = vars_action["_group_actions"]
        for x in var_group_actions:
            if x.dest == arg:
                var_group_actions.remove(x)
                return


def add_alt_arguments(arg_parser):
    arg_parser.add_argument("--python", default="${PYTHON}",
                            help="The path of python interpreter")
    arg_parser.add_argument("--venv",
                            help="Use virtualenv in this directory. "
                                 "Create new virtualenv if not exist. ")


class ArgParserShowUsage(ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        super(ArgParserShowUsage, self).error(message)


def extract_codes_from_model(model_file, dest):
    with open(model_file, "rb") as f:
        read_script(f)
        importer_namespace, envir_info = read_importer_and_source(f)
        # read pickle protocol
        next(pickletools.genops(f))
        # read class name and module
        class_op, class_op_args, _ = next(pickletools.genops(f))
        assert class_op.code == "c", "Invalid model file"
        entrance_module, entrance_class_name = class_op_args.split(" ")
    # write codes
    extract_codes(envir_info["modules"], dest, importer_namespace)
    # write main script
    with open(os.path.join(dest, "parser_.py"), "w") as f:
        f.write("from {} import {}\n".format(
            entrance_module, entrance_class_name))
        f.write("{}.main()\n".format(entrance_class_name))


def main():
    model_file = os.environ.get("MAGIC_MODEL_FILE")

    rest_args = sys.argv[1:]
    if model_file is None:
        arg_parser = ArgParserShowUsage("parser", add_help=False)
        arg_parser.add_argument("--model", metavar="FILE",
                                help="Model file you want to use",
                                required=True)
        args, rest_args = arg_parser.parse_known_args(rest_args)
        model_file = args.model

    arg_parser = ArgParserShowUsage("parser", add_help=False)
    add_alt_arguments(arg_parser)
    arg_parser.add_argument("mode", choices=["predict", "server", "extract", "shell"],
                            help="predict: predict with this model; "
                                 "server: start predict server; "
                                 "extract: extract internal codes of this model;"
                                 "shell: start REPL in this environment")
    args, rest_args = arg_parser.parse_known_args(rest_args)
    mode = args.mode

    if mode == "predict":
        with open(model_file, "rb") as f:
            read_script(f)
            entrance_class = read_until_entrance(f)
        predict_subparser = ArgumentParser("parser predict")
        entrance_class.add_predict_arguments(predict_subparser)
        entrance_class.add_common_arguments(predict_subparser)
        if model_file is not None:
            remove_option(predict_subparser, "--model")
        args = predict_subparser.parse_args(rest_args)
        try:
            # noinspection PyUnresolvedReferences
            from coli.basic_tools.dataclass_argparse import check_argparse_result
            check_argparse_result(args)
        except ImportError:
            pass

        if model_file is not None:
            args.model = model_file
        entrance_class.predict_with_parser(args)
    elif mode == "server":
        with open(model_file, "rb") as f:
            read_script(f)
            entrance_class = read_until_entrance(f)
        server_subparser = ArgumentParser("parser server")
        entrance_class.add_server_arguments(server_subparser)
        if model_file is not None:
            remove_option(server_subparser, "--model")
        args = server_subparser.parse_args(rest_args)
        if model_file is not None:
            args.model = model_file
        entrance_class.load_and_start_server(args)
    elif mode == "extract":
        extract_subparser = ArgumentParser("parser extract")
        extract_subparser.add_argument("dest",
                                       help="destination of extraction")
        args = extract_subparser.parse_args(rest_args)
        extract_codes_from_model(model_file, args.dest)
    elif mode == "shell":
        with open(model_file, "rb") as f:
            read_script(f)
            entrance_class = read_until_entrance(f)
        shell_subparser = ArgumentParser("parser shell")
        entrance_class.add_predict_arguments(shell_subparser)
        entrance_class.add_common_arguments(shell_subparser)
        remove_option(shell_subparser, "--test")
        remove_option(shell_subparser, "--output")
        if model_file is not None:
            remove_option(shell_subparser, "--model")
        args = shell_subparser.parse_args(rest_args)
        parser_class = entrance_class
        parser = parser_class.load(model_file, args)
        try:
            import IPython
            IPython.embed()
        except ModuleNotFoundError:
            import code
            code.interact(local={"parser_class": parser_class,
                                 "parser": parser,
                                 "model_file": model_file})
    else:
        print(f"Invalid mode {mode}")


if __name__ == "__main__":
    main()
