import importlib
import sys
import time
import traceback
import code
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy

from logging import FileHandler

from multiprocessing import Process, Lock
from multiprocessing.pool import ThreadPool

import os
from pprint import pprint

import dataclasses
from dataclasses import is_dataclass

from coli.basic_tools import common_utils
from coli.basic_tools.dataclass_argparse import check_argparse_result
from coli.basic_tools.logger import logger


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
        output = os.path.join(outdir_prefix, "model-" + title)
        if is_dataclass(options_dict):
            options_dict.title = title
            options_dict.output = output
        else:
            options_dict["title"] = title
            options_dict["output"] = output

    if initializer is not None:
        with initializer_lock:
            initializer(options_dict)

    ret = NO_RETURN
    need_reload = False
    need_console = False
    start_time = 0
    cache_keeper = common_utils.cache_keeper or {}

    while ret is NO_RETURN:
        if need_reload:
            from utils.xreload import xreload
            import pathlib
            import gc
            logger.info("Reloading modules...")
            project_root = os.path.abspath(os.path.dirname(__file__) + "/../../") + "/"

            for r_module_name, module in sys.modules.items():
                module_file = getattr(module, "__file__", None)
                if module_file is None:
                    continue
                if module_file.startswith(project_root):
                    if "__main__" not in r_module_name:
                        modify_time = pathlib.Path(module_file).stat().st_mtime
                        if modify_time > start_time:
                            reload_status = xreload(module)
                            if reload_status:
                                logger.info(r_module_name + " updated")

            # do full GC
            try:
                # noinspection PyUnboundLocalVariable
                del dep_parser_class, options, module, exc_info
            except NameError:
                pass
            gc.collect()

        if need_console:
            def exit():
                choice = input("Type yes to exit and start running task: ")
                if choice == "yes":
                    raise SystemExit

            try:
                try:
                    import IPython
                    IPython.embed()
                except ModuleNotFoundError:
                    code.interact(local=locals())
            except SystemExit:
                pass

        start_time = time.time()
        # noinspection PyBroadException
        try:
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
            if getattr(options, "use_exception_handler", False):
                common_utils.cache_keeper = cache_keeper
            ret = func(options)
        except Exception:
            if is_dataclass(options_dict):
                use_exception_handler = options_dict.use_exception_handler
            else:
                use_exception_handler = options_dict.get("use-exception-handler")
            if not use_exception_handler:
                raise
            # handle errors
            traceback.print_exc()
            # waiting for stderr flush
            sys.stderr.flush()
            time.sleep(3)

            # choose error handling method
            need_console = False
            while True:
                # save exc_info to variable to prevent future exception
                exc_info = sys.exc_info()

                def locals_at(frame_level):
                    return list(traceback.walk_tb(exc_info[2]))[frame_level][0].f_locals

                logger.info("PID: {}\n".format(os.getpid()))
                input_cmd = input("Exception occurred. What do you want to do?\n"
                                  "reload - reload codes and retry\n"
                                  "console-keep - keep exception stack and start a interactive console\n"
                                  "extract_tb - print traceback frame summary\n"
                                  "print_local [frame_level] [key] - print local variables in exception stack\n"
                                  "console-reload - clear exception stack, reload, and start a interactive console\n"
                                  "pdb - set trace for pdb \n"
                                  "exit - reraise exception and exit\n"
                                  ">> "
                                  )
                choice, _, args = input_cmd.partition(" ")
                if choice == "exit":
                    raise  # exit the program
                elif choice == "reload":
                    need_reload = True
                    break  # restart running
                elif choice == "console-reload":
                    need_console = True
                    need_reload = True
                    break
                elif choice == "console-keep":
                    def exit():
                        raise SystemExit

                    try:
                        try:
                            import IPython
                            IPython.embed()
                        except ModuleNotFoundError:
                            code.interact(local=locals())
                    except SystemExit:
                        continue  # goto choice
                elif choice == "pdb":
                    import pdb
                    pdb.set_trace()
                elif choice == "extract_tb":
                    pprint(traceback.extract_tb(exc_info[-1]))
                elif choice == "print_local":
                    args_list = args.strip().split(" ")
                    if len(args_list) >= 1:
                        try:
                            frame_level = int(args_list[0])
                        except ValueError:
                            print("print_local [frame_level] [key]")
                            continue
                    if len(args_list) == 1:
                        # noinspection PyUnboundLocalVariable
                        print(locals_at(frame_level).keys())
                    elif len(args_list) == 2:
                        key = args_list[1]
                        print(locals_at(frame_level)[key])
                    else:
                        print("print_local [frame_level] [key]")
                        continue
                else:
                    continue
    return ret


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
        pool = ThreadPool(1)
        for (title, outdir_prefix, mode), options_dict in self.all_options_and_outdirs.items():
            logger.info("Training " + title)
            if self.initializer is not None:
                self.initializer(options_dict)
            # run in a thread to handle KeyboardInterrupt
            result_obj = pool.apply_async(
                lazy_run_parser,
                (self.module_name, self.class_name, title,
                 options_dict, outdir_prefix, None, mode)
            )
            ret = NO_RETURN
            while ret is NO_RETURN:
                try:
                    ret = result_obj.get()
                except KeyboardInterrupt:
                    # handle keyboard interrupt
                    while True:
                        try:
                            answer = input("Really exit ? (yes/no/restart)")
                        except KeyboardInterrupt:
                            continue
                        if answer == "yes" or answer == "restart":
                            async_raise(pool._pool[0].ident, KeyboardInterrupt)
                            pool._pool[0].join()
                            if answer == "yes":
                                raise
                            else:
                                break
                        elif answer == "no":
                            break
            for handler in logger.handlers:
                if isinstance(handler, FileHandler):
                    logger.removeHandler(handler)
            logger.info("{} Done! result is {}".format(title, ret))
        pool.terminate()

    def clear(self):
        self.all_options_and_outdirs = OrderedDict()


class TrainingScheduler(LazyLoadTrainingScheduler):
    """
    Just use it for backward compatibility
    """

    # noinspection PyInitNewSignature
    def __init__(self, parser_class, initializer=None):
        super(TrainingScheduler, self).__init__(parser_class.__module__, parser_class.__name__, initializer)
