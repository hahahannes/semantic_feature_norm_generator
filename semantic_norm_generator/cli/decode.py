import argparse 

from semantic_norm_generator.decoding.run import run_decode

class DecodeCommand():
    def run(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        print('decode')
        run_decode(args)

    def prepare_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument("--answers", dest='answers')
        parser.add_argument("--output", dest='output')
        parser.add_argument("--parallel", dest='parallel', action='store_true')
        parser.add_argument("--lemmatize", dest='lemmatize', action='store_true')
        parser.add_argument("--keep_duplicates_per_concept", dest='keep_duplicates_per_concept', action='store_true')
