import importlib
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy

from logging import FileHandler

from multiprocessing import Process, Lock
from multiprocessing.pool import ThreadPool

import os

from dataclasses import is_dataclass

from coli.basic_tools import common_utils
from coli.basic_tools.dataclass_argparse import check_argparse_result, check_options
from coli.basic_tools.logger import logger
from coli.parser_tools.debug_console import debug_console_wrapper

NO_RETURN = object()


def dict_to_commandline(dic, prefix=()):
    option_cmd = list(prefix)
    for k, v in dic.items():
        assert isinstance(k, str)
        if v is True:
            option_cmd.append("--" + k)
        elif v is False:
            continue
        else:
            option_cmd.append("--" + k)
            if isinstance(v, list):
                option_cmd.extend(str(i) for i in v)
            else:
                option_cmd.append(str(v))

    return option_cmd


def parse_cmd_multistage(dep_parser_class, cmd):
    namespace = Namespace()
    arg_parser = dep_parser_class.get_arg_parser()
    _, rest_cmd = arg_parser.parse_known_args(cmd, namespace)
    stage = 1
    while True:
        next_arg_parser = dep_parser_class.get_next_arg_parser(stage, namespace)
        if next_arg_parser is None:
            if rest_cmd:
                try:
                    from gettext import gettext as _
                except ImportError:
                    def _(message):
                        return message
                msg = _('unrecognized arguments: %s')
                arg_parser.error(msg % ' '.join(rest_cmd))
            else:
                return namespace
        stage += 1
        _, rest_cmd = next_arg_parser.parse_known_args(rest_cmd, namespace)
        arg_parser = next_arg_parser


def parse_dict_multistage(dep_parser_class, dic, prefix=()):
    return parse_cmd_multistage(dep_parser_class, dict_to_commandline(dic, prefix))


def lazy_run_parser(module_name, class_name, title, options_dict, outdir_prefix,
                    initializer_lock, mode="train", initializer=None):
    if mode == "train":
        output = os.path.join(outdir_prefix, title)
        if is_dataclass(options_dict):
            options_dict.title = title
            options_dict.output = output
            check_options(options_dict, mode == "train")
        else:
            options_dict["title"] = title
            options_dict["output"] = output

    if initializer is not None:
        with initializer_lock:
            initializer(options_dict)

    if is_dataclass(options_dict):
        use_exception_handler = options_dict.use_exception_handler
    else:
        use_exception_handler = options_dict.get("use-exception-handler")

    def parse_options_and_run():
        dep_parser_class = getattr(importlib.import_module(module_name), class_name)
        if is_dataclass(options_dict):
            options = options_dict
            funcs = {"train": lambda: dep_parser_class.train_parser,
                     "dev": lambda: dep_parser_class.predict_with_parser,
                     "eval": lambda: dep_parser_class.eval_only}
            func = funcs[mode]()
        else:
            options = parse_dict_multistage(dep_parser_class, options_dict, [mode])
            func = options.func
            check_argparse_result(options)
        return func(options)

    if use_exception_handler:
        common_utils.cache_keeper = {}
        return debug_console_wrapper(parse_options_and_run)
    else:
        return parse_options_and_run()


def async_raise(tid, excobj):
    import ctypes
    import platform
    version_info = platform.sys.version_info
    assert version_info.major >= 3
    # >= python 3.7
    if version_info.minor >= 7:
        tid = ctypes.c_ulong(tid)
    else:
        tid = ctypes.c_long(tid)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(excobj))
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class LazyLoadTrainingScheduler(object):
    """
    Run multiple instance of trainer.
    """

    def __init__(self, module_name, class_name, initializer=None):
        self.module_name = module_name
        self.class_name = class_name
        self.all_options_and_outdirs = OrderedDict()
        self.initializer = initializer

    @classmethod
    def of(cls, parser_class, initializer=None):
        return cls(parser_class.__module__, parser_class.__name__, initializer)

    def add_options(self, title, options_dict, outdir_prefix="", mode="train"):
        if is_dataclass(options_dict):
            options_dict_copy = deepcopy(options_dict)
        else:
            options_dict_copy = dict(options_dict)
        self.all_options_and_outdirs[title, outdir_prefix, mode] = options_dict_copy

    def run_parallel(self):
        initializer_lock = Lock()
        if len(self.all_options_and_outdirs) == 1:
            self.run()
            return

        processes = {}
        for (title, outdir_prefix, mode), options_dict in self.all_options_and_outdirs.items():
            print("Training " + title)
            processes[title, outdir_prefix] = Process(target=lazy_run_parser,
                                                      args=(self.module_name, self.class_name, title,
                                                            options_dict, outdir_prefix, initializer_lock,
                                                            mode, self.initializer)
                                                      )

        try:
            for index, process in processes.items():
                process.start()
            for index, process in processes.items():
                process.join()
        except KeyboardInterrupt:
            for index, process in processes.items():
                process.terminate()

    def run(self):
        for (title, outdir_prefix, mode), options_dict in self.all_options_and_outdirs.items():
            logger.info("Training " + title)
            if self.initializer is not None:
                self.initializer(options_dict)
            ret = lazy_run_parser(self.module_name, self.class_name, title,
                                  options_dict, outdir_prefix, None, mode)
            for handler in logger.handlers:
                if isinstance(handler, FileHandler):
                    logger.removeHandler(handler)
            logger.info("{} Done! result is {}".format(title, ret))

    def clear(self):
        self.all_options_and_outdirs = OrderedDict()


class TrainingScheduler(LazyLoadTrainingScheduler):
    """
    Just use it for backward compatibility
    """

    # noinspection PyInitNewSignature
    def __init__(self, parser_class, initializer=None):
        super(TrainingScheduler, self).__init__(parser_class.__module__, parser_class.__name__, initializer)
