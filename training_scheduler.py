from argparse import ArgumentParser
from collections import OrderedDict
from optparse import OptionParser

from logging import FileHandler

from multiprocessing import Process

from common_utils import parse_dict
from logger import logger


class TrainingScheduler(object):
    """
    Run multiple instance of trainer.
    """
    def __init__(self, train_func, parser, train=None, dev=None, test=None):
        self.train = train
        self.dev = dev
        self.test = test
        self.train_func = train_func
        self.parser = parser
        self.all_options = OrderedDict()

    def add_options(self, title, options_dict, outdir_prefix=""):
        options_dict["title"] = title
        options_dict["outdir"] = outdir_prefix + "model-" + title
        if isinstance(self.parser, OptionParser):
            # For older parser interface
            options, args = parse_dict(self.parser, options_dict)
        else:
            assert isinstance(self.parser, ArgumentParser)
            options = parse_dict(self.parser, options_dict, ["train"])
            self.train_func = options.func
        self.all_options[title] = options

    def run_parallel(self):
        if len(self.all_options) == 1:
            self.run()
            return

        processes = {}
        for title, options in self.all_options.items():
            print("Training " + title)
            processes[title] = Process(target=self.train_func,
                                       args=(options, self.train, self.dev, self.test))

        try:
            for index, process in processes.items():
                process.start()
            for index, process in processes.items():
                process.join()
        except KeyboardInterrupt:
            for index, process in processes.items():
                process.terminate()

    def run(self):
        for title, options in self.all_options.items():
            logger.info("Training " + title)
            self.train_func(options, self.train, self.dev, self.test)
            for handler in logger.handlers:
                if isinstance(handler, FileHandler):
                    logger.removeHandler(handler)