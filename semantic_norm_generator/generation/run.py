from .api.openai import GPTGenerator
from .api.claude import ClaudeGenerator

def run(args):
    model_source = args.model_source
    model = args.model 

    if model_source == 'gpt':
        GPTGenerator(
            args.output_dir,
            args.model,
            args.train_dir,
            args.retrieval_path
        ).run()
    elif model_source == 'claude':
        ClaudeGenerator(
            args.output_dir,
            args.model,
            args.train_dir,
            args.retrieval_path
        ).run()
    elif model_source == 'local':
        pass