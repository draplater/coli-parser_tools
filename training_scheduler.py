import importlib
import sys
import time
import traceback
import weakref
import code
from argparse import Namespace
from collections import OrderedDict

from logging import FileHandler

from multiprocessing import Process, Lock
from multiprocessing.pool import ThreadPool

import os

from logger import logger


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
        options_dict["title"] = title
        options_dict["outdir"] = os.path.join(outdir_prefix, "model-" + title)

    if initializer is not None:
        with initializer_lock:
            initializer(options_dict)

    ret = None
    need_reload = False
    need_console = False
    cache_keeper = {}

    while ret is None:
        if need_reload:
            import gc
            logger.info("Reloading modules...")
            project_root = os.path.dirname(__file__)
            try:
                # noinspection PyUnboundLocalVariable
                del dep_parser_class, options
            except NameError:
                pass
            modules = list(sys.modules.items())
            modules_refs = []
            module = None

            # unload modules in the project
            for r_module_name, module in modules:
                if getattr(module, "__file__", "").startswith(project_root):
                    if "__main__" not in r_module_name:
                        modules_refs.append((r_module_name, weakref.ref(module)))
                        # delete this module instead reload it
                        # because it's difficult to determine reload order
                        del sys.modules[r_module_name]
            del module
            del modules

            # determine whether the module is recycled
            gc.collect()
            for r_module_name, module_ref in modules_refs:
                if module_ref() is not None:
                    logger.info("{} is not collected by gc".format(r_module_name))

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

        # noinspection PyBroadException
        try:
            dep_parser_class = getattr(importlib.import_module(module_name), class_name)
            options = parse_dict_multistage(dep_parser_class, options_dict, [mode])
            if options.use_exception_handler:
                importlib.import_module("common_utils").cache_keeper = cache_keeper
            ret = options.func(options)
        except KeyboardInterrupt:
            # won't happen bacause it's running in a thread
            raise
        except Exception:
            if not options_dict["use-exception-handler"]:
                raise
            # handle errors
            traceback.print_exc()
            # waiting for stderr flush
            sys.stderr.flush()
            time.sleep(3)

            # choose error handling method
            need_console = False
            while True:
                logger.info("PID: {}\n".format(os.getpid()))
                choice = input("Exception occurred. What do you want to do?\n"
                               "reload - reload codes and retry\n"
                               "console-keep - keep exception stack and start a interactive console\n"
                               "console-reload - clear exception stack, reload, and start a interactive console\n"
                               "pdb - set trace for pdb \n"
                               "exit - reraise exception and exit\n"
                               ">> "
                               )
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
                else:
                    continue
    return ret


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
        self.all_options_and_outdirs[title, outdir_prefix, mode] = dict(options_dict)

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
            ret = None
            while ret is None:
                try:
                    ret = result_obj.get()
                except KeyboardInterrupt:
                    # handle keyboard interrupt
                    answer = None
                    while answer != "yes" or answer != "no":
                        try:
                            answer = input("Really exit ? (yes/no)")
                        except KeyboardInterrupt:
                            pass
                    if answer == "yes":
                        continue
                    else:
                        pool.terminate()
                        raise
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
