import argparse 

from semantic_norm_generator.generation import run

class CreateCommand():
    def run(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        run(args)
        
    def prepare_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument("--output_dir", dest='output_dir', help="Path to directory where the file containing the raw completions is saved")
        parser.add_argument("--train_dir", dest='train_dir', help="Path to the directory containing the priming examples. Each file in the directory corresponds to a specific priming")
        parser.add_argument("--retrival_path", dest='retrival_path', help="Path to the file containing the questions")
        parser.add_argument("--model", dest='model', help="Name of the model", choices=["gpt3-davinci", "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301"])
        parser.add_argument("--model_source", dest='model_source', choices=['gpt', 'local', 'claude'], help="Source of the large language model, e.g GPT API or local models from huggingface")
        parser.add_argument("--number_runs", dest='number_runs', help="Number of initializations run. This corresponds to the number of files in the priming directory")