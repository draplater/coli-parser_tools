import code
import os
import sys
import time
import traceback
from pprint import pprint

from coli.basic_tools.logger import logger


NO_RETURN = object()


def debug_console_wrapper(func, *cmd_args, **kwargs):
    ret = NO_RETURN
    need_reload = False
    need_console = False
    start_time = 0

    while ret is NO_RETURN:
        if need_reload:
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
            ret = func(*cmd_args, **kwargs)
        except Exception:
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
                                  "console - keep exception stack and start a interactive console\n"
                                  "extract_tb - print traceback frame summary\n"
                                  "print_local [frame_level] [key] - print local variables in exception stack\n"
                                  "console-reload - clear exception stack, reload, and start a interactive console\n"
                                  "pdb - set trace for pdb \n"
                                  "raise - reraise exception and exit\n"
                                  ">> "
                                  )
                choice, _, args = input_cmd.partition(" ")
                if choice == "raise":
                    raise  # exit the program
                elif choice == "reload":
                    need_reload = True
                    break  # restart running
                elif choice == "console-reload":
                    need_console = True
                    need_reload = True
                    break
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

