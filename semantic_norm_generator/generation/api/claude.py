import os
import anthropic

from .api_generator import APIGenerator

class ClaudeGenerator(APIGenerator):
    def __init__(
        self,
        output_dir,
        model,
        train_dir,
        retrieval_path,
        number_runs,
        number_of_parallel_jobs
    ):
        super().__init__(
            output_dir,
            model,
            train_dir,
            retrieval_path,
            number_runs,
            number_of_parallel_jobs
        )

    def generate_single_prime_sentence(self, train_df, question):
        priming_text = f"{anthropic.HUMAN_PROMPT} Answer the last question. Use the same syntax."
        for row in train_df:
            priming_text += ' ' + row[0]
            priming_text += ' '
            priming_text += row[1]

        text = f"{priming_text} {question}{anthropic.AI_PROMPT}" # See Claude API reference for format
        return text

    def generate_chat_priming_messages(self, train_df, question):
        text = ""
        for row in train_df:
            text += f"{anthropic.HUMAN_PROMPT} {row[0]}"
            text += f"{anthropic.AI_PROMPT} {row[1]}"

        text += f"{anthropic.HUMAN_PROMPT} {question} {anthropic.AI_PROMPT}"
        return text

    def make_request(self, train_df, model, question):
        client = anthropic.Client(os.environ['ANTHROPIC_API_KEY'])
        prompt = self.generate_single_prime_sentence(train_df, question)

        response = client.completion(
            prompt=prompt,
            stop_sequences = [anthropic.HUMAN_PROMPT],
            model=model,
            max_tokens_to_sample=80,
        )
        return response['completion']