import code
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pprint import pprint

from coli.basic_tools.logger import logger
from coli.basic_tools.timeout import Timeout

NO_RETURN = object()


@dataclass
class HandlerResult(object):
    need_reload: bool = False
    need_console: bool = False


def handle_exception(frames_and_linenos, exc=None, tb=None) -> HandlerResult:
    while True:
        # save exc_info to variable to prevent future exception

        def locals_at(frame_level):
            return frames_and_linenos[frame_level][0].f_locals

        logger.info("PID: {}\n".format(os.getpid()))
        try:
            with Timeout(300):
                input_cmd = input("What do you want to do?\n"
                                  "(You have 10 minutes to decide.)\n"
                                  "reload - reload codes and retry\n"
                                  "console - keep exception stack and start a interactive console\n"
                                  "print_stack - print traceback frame summary\n"
                                  "print_local [frame_level] [key] - print local variables in exception stack\n"
                                  "console-reload - clear exception stack, reload, and start a interactive console\n"
                                  "pdb - set trace for pdb\n"
                                  "continue - just let it go\n"
                                  ">> "
                                  )
        except TimeoutError:
            input_cmd = "raise"
        choice, _, args = input_cmd.partition(" ")
        if choice == "continue":
            return HandlerResult(False, False)
        elif choice == "reload":
            return HandlerResult(need_reload=True)  # restart running
        elif choice == "console-reload":
            return HandlerResult(need_reload=True, need_console=True)
        elif choice == "console":
            args_list = args.strip().split(" ")
            if len(args_list) == 1:
                try:
                    frame_level = int(args_list[0])
                except ValueError:
                    print("console [frame_level]")
                    continue
            else:
                frame_level = -1

            try:
                try:
                    from IPython.terminal.embed import InteractiveShellEmbed
                    InteractiveShellEmbed().mainloop(local_ns=locals_at(frame_level))
                except ModuleNotFoundError:
                    code.interact(local=locals_at(frame_level))
            except SystemExit:
                continue  # goto choice
        elif choice == "pdb":
            import pdb
            if tb:
                pdb.post_mortem(tb)
            else:
                # TODO: still some bugs
                debugger = pdb.Pdb()
                debugger.set_trace()
        elif choice == "print_stack":
            traceback.print_list(traceback.StackSummary.extract(frames_and_linenos))
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
    return HandlerResult(need_reload=False, need_console=False)


def handle_keyboard_interrupt(sig_no, frame):
    frames = list(traceback.walk_stack(frame))
    frames.reverse()
    stack = traceback.StackSummary.extract(frames)
    traceback.print_list(stack)
    handle_exception(frames)


def debug_console_wrapper(func, *cmd_args, **kwargs):
    ret = NO_RETURN
    handler_ret = HandlerResult()
    start_time = 0

    while ret is NO_RETURN:
        if handler_ret.need_reload:
            from .xreload import xreload
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
                del module, exc_info
            except NameError:
                pass
            gc.collect()

        if handler_ret.need_console:
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
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handle_keyboard_interrupt)
        # noinspection PyBroadException
        try:
            ret = func(*cmd_args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            # handle errors
            traceback.print_exc()
            # waiting for stderr flush
            sys.stderr.flush()
            time.sleep(1)

            # choose error handling method
            handler_ret = handle_exception(list(traceback.walk_tb(exc_info[2])),
                                           exc_info[1], exc_info[2])
            if not handler_ret.need_reload and not handler_ret.need_console:
                raise
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)
    return ret
