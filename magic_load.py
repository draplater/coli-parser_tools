#!/usr/bin/env python3
import os
import sys
import tempfile
import pickle
import struct
import base64
from argparse import ArgumentParser
from contextlib import contextmanager

magic_header = b'#!/bin/sh\n# MAGIC_STRING = SUGAR_RUSH\nSCRIPT_SIZE='
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
        raise TypeError("It's not a magic pack!")
    script_size_packed = reader.read(script_size_len)
    script_size = struct.unpack("!Q", base64.b64decode(script_size_packed))[0]
    rest = reader.read(script_size - len(magic_header) - script_size_len)
    return header + script_size_packed + rest


def read_importer_and_source(reader):
    magic_importer_source, module_sources = pickle.load(reader)
    importer_namespace = {}
    exec(magic_importer_source, importer_namespace)
    return importer_namespace, module_sources


def read_entrance(reader, importer_namespace, module_sources):
    importer = importer_namespace["MagicPackImporter"](module_sources)
    importer.install()
    entrance_class = pickle.load(reader)
    return entrance_class


def read_until_entrance(reader):
    importer_namespace, module_sources = read_importer_and_source(reader)
    return read_entrance(reader, importer_namespace, module_sources)


def extract_codes(all_sources, importer_namespace, dest):
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
                filename = importer_namespace["_path_split"](original_file)[1]
                full_path = os.path.join(package_path, filename)
                os.makedirs(package_path, exist_ok=True)
                with open(full_path, "wb") as f:
                    f.write(source)
                importer_namespace["restore_assets"](package_path, assets)
            elif module_info[0] == "cython_source":
                sources, assets = module_info[1:]
                importer_namespace["extract_cython_files"](
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
        if (vars(action)['option_strings']
            and vars(action)['option_strings'][0] == arg) \
                or vars(action)['dest'] == arg:
            parser._remove_action(action)

    for action in parser._action_groups:
        vars_action = vars(action)
        var_group_actions = vars_action['_group_actions']
        for x in var_group_actions:
            if x.dest == arg:
                var_group_actions.remove(x)
                return


def add_alt_arguments(arg_parser):
    arg_parser.add_argument('--python', default="${PYTHON}",
                            help="The path of python interpreter")
    arg_parser.add_argument('--venv',
                            help="Use virtualenv in this directory. "
                                 "Create new virtualenv if not exist. ")


class ArgParserShowUsage(ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        super(ArgParserShowUsage, self).error(message)


if __name__ == "__main__":
    # this file can be inserted in to a shell script
    # and the "$xxx" will be replaced into real value
    model_file = "$0"

    if __file__.startswith(tempfile.gettempdir()):
        os.remove(__file__)

    rest_args = sys.argv[1:]
    # "\x24\x30" == "$0", escape shell substitution
    if model_file == "\x24\x30":
        arg_parser = ArgParserShowUsage("parser", add_help=False)
        arg_parser.add_argument("--model", metavar="FILE",
                                help="Model file you want to use",
                                required=True)
        args, rest_args = arg_parser.parse_known_args(rest_args)
        model_file = args.model

    arg_parser = ArgParserShowUsage("parser", add_help=False)
    add_alt_arguments(arg_parser)
    arg_parser.add_argument("mode", choices=["predict", "extract", "shell"],
                            help="predict: predict with this model; "
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
        if model_file != '\x24\x30':
            remove_option(predict_subparser, "--model")
        args = predict_subparser.parse_args(rest_args)
        try:
            from coli.basic_tools.dataclass_argparse import check_argparse_result

            check_argparse_result(args)
        except ImportError:
            pass

        if model_file != '\x24\x30':
            args.model = model_file
            entrance_class.predict_with_parser(args)
    elif mode == "extract":
        with open(model_file, "rb") as f:
            read_script(f)
            importer_namespace, envir_info = read_importer_and_source(f)
        extract_subparser = ArgumentParser("parser extract")
        extract_subparser.add_argument("dest",
                                       help="destination of extraction")
        args = extract_subparser.parse_args(rest_args)
        extract_codes(envir_info["modules"], importer_namespace, args.dest)
    elif mode == "shell":
        with open(model_file, "rb") as f:
            read_script(f)
            entrance_class = read_until_entrance(f)
        import code

        code.interact(local={"Parser": entrance_class})
    else:
        print(f"Invalid mode {mode}")
