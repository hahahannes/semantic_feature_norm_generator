from .api.openai import GPTGenerator
from .api.claude import ClaudeGenerator
from .local.llm_generator import LLMGenerator

def run(args):
    model_source = args.model_source
    model = args.model 

    if model_source == 'gpt':
        GPTGenerator(
            args.output_dir,
            args.model,
            args.train_dir,
            args.retrieval_path,
            args.number_runs
        ).run()
    elif model_source == 'claude':
        ClaudeGenerator(
            args.output_dir,
            args.model,
            args.train_dir,
            args.retrieval_path,
            args.number_runs
        ).run()
    elif model_source == 'local':
        LLMGenerator(
            args.output_dir,
            args.model,
            args.train_dir,
            args.retrieval_path,
            args.number_runs
        ).run()