import argparse 

from semantic_norm_generator.decoding.run import run_decode

class DecodeCommand():
    def run(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        run_decode(args)

    def prepare_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument("--answers", dest='answers', help="Path to the file containing the raw completions")
        parser.add_argument("--output", dest='output', help="Directory to put the file containing decoded features")
        parser.add_argument("--number_of_parallel_jobs", dest="number_of_parallel_jobs", help="Number of cores used for generation. Especially useful when APIs are used", default=2, type=int)        parser.add_argument("--lemmatize", dest='lemmatize', action='store_true', help="Lemmatize the feature norm")
        parser.add_argument("--keep_duplicates_per_concept", dest='keep_duplicates_per_concept', action='store_true', help="Keep duplicated features per concept")
