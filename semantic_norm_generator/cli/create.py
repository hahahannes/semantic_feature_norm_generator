import argparse 

from semantic_norm_generator.generation.run import run

class CreateCommand():
    def run(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        run(args)
        
    def prepare_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument("--output_dir", dest='output_dir', help="Path to directory where the file containing the raw completions is saved")
        parser.add_argument("--train_dir", dest='train_dir', help="Path to the directory containing the priming examples. Each file in the directory corresponds to a specific priming")
        parser.add_argument("--retrieval_path", dest='retrieval_path', help="Path to the file containing the questions")
        parser.add_argument("--model", dest='model', help="Name of the model e.g. gpt3-davinci or path to local model weights e.g ~/falcon-65b") #choices=["gpt3-davinci", "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301", "claude-1", "claude-2", "falcon-40b"]
        parser.add_argument("--model_source", dest='model_source', choices=['gpt', 'local', 'claude'], help="Source of the large language model, e.g GPT API or local models from huggingface")
        parser.add_argument("--number_runs", dest='number_runs', help="Number of initializations run. This corresponds to the number of files in the priming directory", type=int)
        parser.add_argument("--number_of_parallel_jobs", dest="number_of_parallel_jobs", help="Number of cores used for generation. Especially useful when APIs are used", default=2, type=int)
