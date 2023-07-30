from .api_generator import APIGenerator

import openai

class GPTGenerator(APIGenerator):
    def __init__(self):
        super().__init__()

    def generate_chat_priming_messages(self, train_df, question):
        questions = []
        priming = []
        for row in train_df:
            questions.append(row[1])
            priming.append(row[2])
            
        questions.append(question)

        messages = [{
                "role": "user",
                "content": questions[0]
            }, 
            {
                "role": "assistant",
                "content": priming[0]
            }, 
            {
                "role": "user",
                "content": questions[1]
            }, 
            {
                "role": "assistant",
                "content": priming[1]
            }, 
            {
                "role": "user",
                "content": questions[2]
            }, 
            {
                "role": "assistant",
                "content": priming[2]
            }, 
            
            {
                "role": "user",
                "content": questions[3]
            },
        ]
        return messages

    def make_request(self, train_df, model, question):
        if model == 'gpt-3.5-turbo' or model == 'gpt-3.5-turbo-0301' or model == 'gpt-4':
            messages = generate_chat_priming_messages(train_df, question)
            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.5, frequency_penalty=0.33, max_tokens=70, n=1)
            response_text = response['choices'][0]['message']['content']
        else:
            priming_text = generate_single_prime_sentence(train_df, question)
            response = openai.Completion.create(engine=model, prompt=priming_text, temperature=0.5, frequency_penalty=0.33, max_tokens=70, n=1)
            response_text = response['choices'][0]['text']
        return response_text
