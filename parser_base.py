import argparse
import pickle
import random
from typing import Generic, TypeVar, Type, Dict, List, Optional, Iterable

from io import open

from argparse import ArgumentParser

import os
import sys
import subprocess
from abc import ABCMeta, abstractmethod

import time

from dataclasses import dataclass

from coli.basic_tools.dataclass_argparse import REQUIRED, argfield, DataClassArgParser, check_argparse_result, \
    pretty_format, OptionsBase
from coli.basic_tools.common_utils import set_proc_name, ensure_dir, smart_open, NoPickle, cache_result
from coli.basic_tools.logger import get_logger, default_logger, log_to_file
from coli.data_utils.dataset import DataFormatBase
from coli.parser_tools.debug_console import debug_console_wrapper
from coli.parser_tools.magic_load import remove_option
from coli.parser_tools.magic_pack import get_codes
from coli.parser_tools.training_scheduler import parse_cmd_multistage
from coli.basic_tools.base_service import WebAPIService

DF = TypeVar("DF", bound=DataFormatBase)


class DependencyParserBase(Generic[DF], metaclass=ABCMeta):
    DataType: Type[DF] = None
    available_data_formats: Dict[str, Type[DF]] = {}
    default_data_format_name = "default"

    def __init__(self, options, data_train=None, *args, **kwargs):
        super(DependencyParserBase, self).__init__()
        self.options = options
        # do not log to console if not training
        self.log_to_file = NoPickle(data_train is not None)

    @property
    def logger(self):
        if getattr(self, "_logger", None) is None:
            self._logger = NoPickle(
                self.get_logger(self.options,
                                log_to_file=self.log_to_file))
        return self._logger

    @classmethod
    def get_data_formats(cls) -> Dict[str, Type[DF]]:
        """ for old class which has "DataType" but not "available_data_formats" """
        if not cls.available_data_formats:
            return {"default": cls.DataType}
        else:
            return cls.available_data_formats

    @property
    def data_format_class(self) -> Type[DF]:
        return self.get_data_formats()[self.options.data_format]

    @abstractmethod
    def train(self, graphs: List[DF], *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, graphs):
        """:rtype: list[self.DataType]"""
        pass

    @abstractmethod
    def save(self, prefix):
        pass

    @classmethod
    @abstractmethod
    def load(cls, prefix, new_options=None):
        pass

    @dataclass
    class Options(OptionsBase):
        title: str = argfield("default",
                              help="Name of this task")
        train: str = argfield(metavar="FILE",
                              help="Path of training set")
        dev: List[str] = argfield(metavar="FILE", nargs="+",
                                  help="Path of development set")
        max_save: int = argfield(100,
                                 help="keep only best n model when training")
        epochs: int = argfield(30,
                               help="Training epochs")
        debug_cache: bool = argfield(False,
                                     help="Use cache file for quick debugging")

        # both train and predict
        output: str = argfield(predict_time=True, predict_default=REQUIRED,
                               help="Output path")
        test: Optional[str] = argfield(default=None,
                                       metavar="FILE", predict_time=True, predict_default=REQUIRED,
                                       help="Path of test set")
        model: str = argfield(default="model.", help="Load/Save model file", metavar="FILE",
                              predict_time=True, predict_default=REQUIRED)
        dynet_seed: int = argfield(42, predict_time=True)
        dynet_autobatch: int = argfield(0, predict_time=True)
        dynet_mem: int = argfield(0, predict_time=True)
        dynet_gpus: int = argfield(0, predict_time=True)
        dynet_l2: float = argfield(0.0, predict_time=True)
        weight_decay: float = argfield(0.0, predict_time=True)
        output_scores: bool = argfield(False, predict_time=True)
        data_format: str = argfield("default", predict_time=True,
                                    help="format of input data")
        # ???
        # group.add_argument("--data-format", dest="data_format",
        #                    choices=cls.get_data_formats(),
        #                    default=cls.default_data_format_name)
        bilm_cache: str = argfield(None, metavar="FILE", predict_time=True,
                                   help="path of elmo cache file")
        bilm_use_cache_only: bool = argfield(
            False, predict_time=True,
            help="use elmo in cache file only, do not generate new elmo")
        bilm_path: Optional[str] = argfield(None, metavar="FILE", predict_time=True,
                                            help="path of elmo model")
        bilm_stateless: bool = argfield(False, predict_time=True,
                                        help="only use stateless elmo")
        bilm_gpu: str = argfield("", predict_time=True,
                                 help="run elmo on these gpu")
        use_exception_handler: bool = argfield(
            False, predict_default=True, predict_time=True,
            help="useful tools for quick debugging when encountering an error")

        # predict only
        eval: bool = argfield(predict_default=False, train_time=False, predict_time=True)
        input_format: str = argfield(
            choices=["standard", "tokenlist",
                     "space", "english", "english-line"],
            help='Input format. (default)"standard": use the same format of treebank;\n'
                 'tokenlist: like [[(sent_1_word1, sent_1_pos1), ...], [...]];\n'
                 'space: sentence is separated by newlines, and words are separated by space;'
                 'no POSTag info will be used. \n'
                 'english: raw english sentence that will be processed by NLTK tokenizer, '
                 'no POSTag info will be used.',
            predict_time=True, train_time=False,
            predict_default="standard"
        )

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        DataClassArgParser("", arg_parser, {"default": cls.Options()}, mode="train")

    @classmethod
    def add_predict_arguments(cls, arg_parser):
        DataClassArgParser(
            "", arg_parser,
            {"default": cls.Options().to_predict_default()},
            mode="predict")

    @classmethod
    def add_common_arguments(cls, arg_parser):
        pass

    @classmethod
    def options_hook(cls, options):
        pass

    def get_log_file(self, options):
        if getattr(self, "logger_timestamp", None) is None:
            self.logger_timestamp = int(time.time())
        return os.path.join(
            options.output, f"{self.logger_timestamp}_train.log")

    def get_logger(self, options, log_to_console=True, log_to_file=True):
        return get_logger(files=self.get_log_file(options) if log_to_file else None,
                          log_to_console=log_to_console,
                          name=getattr(options, "title", "logger"))

    def write_result(self, output_file: str, data: Iterable[DF]):
        if hasattr(self.data_format_class, "write_to_file"):
            self.data_format_class.write_to_file(output_file, data)
        else:
            with open(output_file, "w") as f:
                for output in data:
                    f.write(output.to_string())

    @classmethod
    def train_parser(cls, options, data_train=None, data_dev=None, data_test=None):
        if sys.platform.startswith("linux"):
            set_proc_name(options.title)
        default_logger.name = options.title
        ensure_dir(options.output)

        cls.options_hook(options)
        DataFormatClass = cls.get_data_formats()[options.data_format]

        @cache_result(options.output + "/" + "input_data_cache.pkl",
                      enable=options.debug_cache)
        def load_data(data_train, data_dev, data_test):
            if data_train is None:
                data_train = DataFormatClass.from_file(options.train)

            if data_dev is None:
                data_dev = {i: DataFormatClass.from_file(i, False) for i in options.dev}

            if data_test is None and options.test is not None:
                data_test = DataFormatClass.from_file(options.test, False)
            else:
                data_test = None
            return data_train, data_dev, data_test

        data_train, data_dev, data_test = load_data(data_train, data_dev, data_test)

        if options.bilm_cache is not None:
            if not os.path.exists(options.bilm_cache):
                train_sents = set(tuple(sent.words) for sent in data_train)
                dev_sentences = set()
                for one_data_dev in data_dev.values():
                    dev_sentences.update(set(tuple(sent.words) for sent in one_data_dev))
                if data_test is not None:
                    dev_sentences.update(set(tuple(sent.words) for sent in data_test))
                dev_sentences -= train_sents
                default_logger.info("Considering {} training sentences and {} dev sentences for bilm cache".format(
                    len(train_sents), len(dev_sentences)))
                # avoid running tensorflow in current process
                script_path = os.path.join(os.path.dirname(__file__), "../../bilm/cache_manager.py")
                p = subprocess.Popen([sys.executable, script_path, "pickle"], stdin=subprocess.PIPE, stdout=sys.stdout,
                                     stderr=sys.stderr)
                args = (options.bilm_path, options.bilm_cache, train_sents, dev_sentences, options.bilm_gpu)
                p.communicate(pickle.dumps(args))
                # pickle.dump(args, p.stdin)
                if p.returncode != 0:
                    raise Exception("Error when generating bilm cache.")
        try:
            os.makedirs(options.output)
        except OSError:
            pass

        return cls.repeat_train_and_validate(
            data_train, data_dev, data_test, options)

    @classmethod
    def repeat_train_and_validate(cls, data_train, data_devs, data_test, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        # noinspection PyArgumentList
        parser = cls(options, data_train)
        log_to_file(parser.get_log_file(options))
        parser.logger.info('Options:\n%s', pretty_format(options.__dict__))
        random_obj = random.Random(1)
        for epoch in range(options.epochs):
            parser.logger.info('Starting epoch %d', epoch)
            random_obj.shuffle(data_train)
            parser.train(data_train)

            # save model and delete old model
            for i in range(0, epoch - options.max_save):
                path = os.path.join(options.output, os.path.basename(options.model)) + str(i + 1)
                if os.path.exists(path):
                    os.remove(path)
            path = os.path.join(options.output, os.path.basename(options.model)) + str(epoch + 1)
            parser.save(path)

            def predict(sentences, gold_file, output_file):
                result_itr = parser.predict(sentences)
                parser.write_result(output_file, result_itr)
                DataFormatClass.evaluate_with_external_program(gold_file, output_file)

            for file_name, file_content in data_devs.items():
                dev_output = cls.get_output_name(options.output, file_name, epoch)
                predict(file_content, file_name, dev_output)

    @classmethod
    def get_output_name(cls, out_dir, file_name, epoch):
        return f"{out_dir}/{epoch}.out"

    @classmethod
    def load_with_options(cls, options):
        default_logger.info('Loading Model...')
        parser = cls.load(options.model, options)
        parser.logger.info('Model loaded')
        return parser

    @classmethod
    def get_test_data(cls, parser, options):
        DataFormatClass = cls.get_data_formats()[parser.options.data_format]
        if options.input_format == "standard":
            data_test = DataFormatClass.from_file(options.test, False)
        elif options.input_format == "space":
            with smart_open(options.test) as f:
                data_test = [DataFormatClass.from_words_and_postags(
                    [(word if word else " ", "X") for word in line.strip().split(" ")])
                    for line in f]
        elif options.input_format.startswith("english"):
            from nltk import download, sent_tokenize
            from nltk.tokenize import TreebankWordTokenizer
            download("punkt")
            with smart_open(options.test) as f:
                raw_sents = []
                for line in f:
                    if options.input_format == "english-line":
                        raw_sents.append(line.strip())
                    else:
                        this_line_sents = sent_tokenize(line.strip())
                        raw_sents.extend(this_line_sents)
                tokenized_sents = TreebankWordTokenizer().tokenize_sents(raw_sents)
                data_test = [DataFormatClass.from_words_and_postags([(token, "X") for token in sent])
                             for sent in tokenized_sents]
        elif options.input_format == "tokenlist":
            with smart_open(options.test) as f:
                items = eval(f.read())
            data_test = [DataFormatClass.from_words_and_postags(item) for item in items]
        else:
            raise ValueError("invalid format option")
        return data_test

    def from_strings(self, inputs, input_format="standard"):
        DataFormatClass = self.get_data_formats()[self.options.data_format]
        if input_format == "standard":
            data_test = [DataFormatClass.from_string(i) for i in inputs]
        elif input_format == "words":
            data_test = [DataFormatClass.from_words_and_postags(
                [(word if word else " ", "X") for word in words])
                for words in inputs]
        elif input_format == "words_and_postags":
            data_test = [DataFormatClass.from_words_and_postags(i) for i in inputs]
        elif input_format.startswith("english"):
            from nltk import download, sent_tokenize
            from nltk.tokenize import TreebankWordTokenizer
            download("punkt")
            tokenizer = TreebankWordTokenizer()
            data_test = [
                DataFormatClass.from_words_and_postags(
                    [(token, "X") for token in tokenizer.tokenize(sent)])
                for sent in inputs]
        else:
            raise ValueError("invalid format option")

        return [i.to_string() for i in self.predict(data_test)]

    def dispatch_service(self, base_service, name=None):
        """
        :type base_service: WebAPIService
        """
        if name is None:
            name = self.__class__.__name__

        from jsonrpc.backend.flask import JSONRPCAPI
        parser_api = JSONRPCAPI()
        parser_api.dispatcher.add_method(self.from_strings)

        def view_func_wrap():
            return parser_api.jsonrpc()

        view_func_wrap.__name__ = name

        base_service.add_url_rule(
            "/api/" + name, view_func=view_func_wrap, methods=["POST"])

    def start_server(self, name=None, host="0.0.0.0", port=9999):
        service = WebAPIService(name=name or self.__class__.__name__)
        self.dispatch_service(service, name)
        service.run(host, port)

    @classmethod
    def load_and_start_server(cls, options):
        parser = cls.load_with_options(options)
        parser.logger.info("Starting Server...")
        parser.start_server(options.api_name, options.host, options.port)

    @classmethod
    def add_server_arguments(cls, server_subparser):
        cls.add_predict_arguments(server_subparser)
        cls.add_common_arguments(server_subparser)
        remove_option(server_subparser, "--test")
        remove_option(server_subparser, "--output")
        remove_option(server_subparser, "--eval")
        server_subparser.add_argument("--api-name")
        server_subparser.add_argument("--host", default="0.0.0.0")
        server_subparser.add_argument("--port", type=int, default=9995)
        server_subparser.set_defaults(func=cls.load_and_start_server)

    @classmethod
    def predict_with_parser(cls, options):
        parser = cls.load_with_options(options)
        ts = time.time()
        data_test = cls.get_test_data(parser, options)
        parser.write_result(options.output, parser.predict(data_test))
        te = time.time()
        parser.logger.info('Finished predicting and writing test. %.2f seconds.', te - ts)

        if options.eval:
            DataFormatClass = cls.get_data_formats()[parser.options.data_format]
            DataFormatClass.evaluate_with_external_program(options.test,
                                                           options.output)

    @classmethod
    def get_arg_parser(cls):
        parser = ArgumentParser(sys.argv[0])
        cls.fill_arg_parser(parser)
        return parser

    @classmethod
    def resave(cls, options):
        parser = cls.load_with_options(options)
        parser.codes = get_codes(os.path.join(os.path.dirname(__file__), "../../"))
        parser.save(options.output)

    @classmethod
    def fill_arg_parser(cls, parser):
        sub_parsers = parser.add_subparsers()
        sub_parsers.required = True
        sub_parsers.dest = 'mode'

        # Train
        # train_subparser = sub_parsers.add_parser("train")
        # cls.add_parser_arguments(train_subparser)
        # cls.add_common_arguments(train_subparser)
        # train_subparser.set_defaults(func=cls.train_parser)

        # Predict
        predict_subparser = sub_parsers.add_parser("predict")
        cls.add_predict_arguments(predict_subparser)
        cls.add_common_arguments(predict_subparser)
        predict_subparser.set_defaults(func=cls.predict_with_parser)

        # Resave
        resave_subparser = sub_parsers.add_parser("resave")
        cls.add_predict_arguments(resave_subparser)
        cls.add_common_arguments(resave_subparser)
        remove_option(resave_subparser, "--test")
        remove_option(resave_subparser, "--eval")
        resave_subparser.set_defaults(func=cls.resave)

        # Server
        server_subparser = sub_parsers.add_parser("server")
        cls.add_server_arguments(server_subparser)

        eval_subparser = sub_parsers.add_parser("eval")
        eval_subparser.add_argument("--data-format", dest="data_format",
                                    choices=cls.get_data_formats(),
                                    default=cls.default_data_format_name)
        eval_subparser.add_argument("gold")
        eval_subparser.add_argument("system")
        eval_subparser.set_defaults(func=cls.eval_only)

    @classmethod
    def get_training_scheduler(cls):
        from coli.parser_tools.training_scheduler import TrainingScheduler
        return TrainingScheduler(cls)

    @classmethod
    def eval_only(cls, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        DataFormatClass.evaluate_with_external_program(options.gold, options.system)

    @classmethod
    def get_next_arg_parser(cls, stage, options):
        return None

    @classmethod
    def fill_missing_params(cls, options):
        test_arg_parser = ArgumentParser()
        cls.add_parser_arguments(test_arg_parser)
        cls.add_common_arguments(test_arg_parser)
        # noinspection PyUnresolvedReferences
        for action in test_arg_parser._actions:
            if action.default != argparse.SUPPRESS:
                if getattr(options, action.dest, None) is None:
                    default_logger.info(
                        "Add missing option: {}={}".format(action.dest, action.default))
                    setattr(options, action.dest, action.default)

    @classmethod
    def main(cls, argv=None):
        if argv is None:
            argv = sys.argv[1:]
        args = parse_cmd_multistage(cls, argv)
        check_argparse_result(args)
        if args.use_exception_handler:
            debug_console_wrapper(args.func, args)
        else:
            args.func(args)
