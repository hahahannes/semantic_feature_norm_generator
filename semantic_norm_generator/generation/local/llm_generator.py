import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from os.path import join as pjoin
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

from semantic_norm_generator.generation.generation_jobs import yield_generation_jobs
from semantic_norm_generator.generation.priming import generate_single_prime_sentence

class LLMGenerator():
    def __init__(
        self,
        output_dir,
        model_name_or_path,
        train_dir,
        retrieval_path,
        number_runs
    ):
        self.train_dir = train_dir
        self.retrieval_path = retrieval_path
        self.output_dir = output_dir
        self.raw_feature_path = pjoin(output_dir, 'raw_completions.csv')
        self.number_runs = number_runs

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number GPUs: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")

        print("load tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )

        print("load model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True, 
            device_map="auto", 
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            #torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    def run(self):
        print("Run Generation")

        self.initialize_output_file()
        
        start = time.time()

        with open(self.raw_feature_path, 'a') as f:
            for job in yield_generation_jobs(self.raw_feature_path, self.train_dir, self.retrieval_path, self.number_runs):
                priming = job['priming']
                question = job['question']
                concept = job['concept']
                concept_id = job['concept_id']
                run_nr = job['run_nr']

                text = generate_single_prime_sentence(priming, question)
                #"Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
                
                sequences = self.pipeline(
                    text,
                    max_new_tokens=70,
                    do_sample=True,
                    temperature=0.5,
                    top_p=1.0,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                end = time.time()
                print(end-start)

                answer = sequences[0]['generated_text']
                print(f"Result: {answer}")

                text = f'{concept},"{answer}",{concept_id},{run_nr}'
                f.write(text + '\n')
                f.flush()

    def initialize_output_file(self):
        if not os.path.exists(self.raw_feature_path):
            with open(self.raw_feature_path, "w") as out_file:
                out_file.write("concept,answer,concept_id,run_nr")
                out_file.flush()
