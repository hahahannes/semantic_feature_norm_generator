import argparse 

from semantic_norm_generator.generation.create_gpt_feature_norm import run

class CreateCommand():
    def run(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        print('create')
        run(args)
        
    def prepare_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument("--output_dir", dest='output_dir')
        parser.add_argument("--train_dir", dest='train_dir')
        parser.add_argument("--retrival_path", dest='retrival_path')
        parser.add_argument("--model", dest='model')
        parser.add_argument("--model_source", dest='model_source')
        parser.add_argument("--number_runs", dest='number_runs')