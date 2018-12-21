#!/usr/bin/env python3
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--python', default="${PYTHON}",
                        help="The path of python interpreter")
    parser.add_argument('--venv',
                        help="Use virtualenv in this directory. "
                             "Create new virtualenv if not exist. ")
    args, rest = parser.parse_known_args()
    if args.venv is None or os.environ.get("USE_VENV") != "true":
        print(args.python)
    else:
        import os
        import sys

        path_exists = os.path.exists(args.venv)
        if path_exists and not os.path.isdir(args.venv):
            print(f"{args.venv} is not a directory", file=sys.stderr)
            sys.exit(1)

        if sys.platform == 'win32':
            bin_name = "Scripts"
            file_name = "python.exe"
        else:
            bin_name = 'bin'
            file_name = "python3"
        python_path = os.path.join(args.venv, bin_name, file_name)
        if os.path.exists(python_path):
            print(python_path)
        else:
            import venv

            if path_exists and os.listdir(args.venv):
                print(f"{args.venv} is neither a virtualenv nor a empty directory",
                      file=sys.stderr)
                sys.exit(1)
            # create new venv
            venv.create(args.venv, with_pip=True)
            if os.path.exists(python_path):
                print(python_path)
            else:
                print(f"failed to create virtualenv", file=sys.stderr)
                sys.exit(1)
