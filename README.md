# Installation
```
git clone https://github.com/hahahannes/semantic_feature_norm_generator
cd semantic_feature_norm_generator
pip install -e .
```

# Usage
The following examples use files from the `doc` directory. Either change the path or run the examples in this directory.

## Generation
The `model` parameter corresponds either to the model name from an API or the name of Huggingface model or the path to weights downloaded from Huggingface. Possible model names from OPENAI can be found here: https://platform.openai.com/docs/models/overview. For example use `davinci` if you want to use the GPT-3 davinci model. 

Note, that the files in the `train_dir` directory need to be named `train_{i}.csv` where `i` specifies the run number. It is recommended to use multiple priming initializations to create more diverse features. Counting starts with `1`. For example, create two files `train_1.csv` and `train_2.csv` if you want to create features with two different primings.

```bash
usage: python -m semantic_norm_generator create [-h] [--output_dir OUTPUT_DIR] [--train_dir TRAIN_DIR] [--retrival_path RETRIVAL_PATH]
                          [--model {gpt3-davinci,gpt-3.5-turbo,gpt-4,gpt-3.5-turbo-0301}] [--model_source {gpt,local,claude}] [--number_runs NUMBER_RUNS]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Path to directory where the file containing the raw completions is saved
  --train_dir TRAIN_DIR
                        Path to the directory containing the priming examples. Each file in the directory corresponds to a specific priming
  --retrieval_path RETRIVAL_PATH
                        Path to the file containing the questions
  --model {gpt3-davinci,gpt-3.5-turbo,gpt-4,gpt-3.5-turbo-0301}
                        Name of the model
  --model_source {gpt,local,claude}
                        Source of the large language model, e.g GPT API or local models from huggingface
  --number_runs NUMBER_RUNS
                        Number of initializations run. This corresponds to the number of files in the priming directory
```

### Example
```
python -m semantic_norm_generator create --output_dir=. --train_dir=doc/priming --retrieval_path=doc/concepts.csv --model_source=gpt --model=davinci --number_runs=1 --number_of_parallel_jobs=2
```


### API Keys
If you want to use commercial APIs, e.g. from OpenAI or Anthropic, you need to have an API Key and make it accessible to the tool.
For example by setting the environment variable `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
```
OPENAI_API_KEY=key python -m semantic_norm_generator --help
```

## Decoding
```bash
usage: python -m semantic_norm_generator decode [-h] [--answers ANSWERS] [--output OUTPUT] [--parallel] [--lemmatize] [--keep_duplicates_per_concept]

optional arguments:
  -h, --help            show this help message and exit
  --answers ANSWERS     Path to the file containing the raw completions
  --output OUTPUT       Directory to put the file containing decoded features
  --parallel            Run in parallel
  --lemmatize           Lemmatize the feature norm
  --keep_duplicates_per_concept
                        Keep duplicated features per concept
```
### Example
```
python -m semantic_norm_generator decode --answers=raw_feature_norm_from_gpt.csv --output=. --parallel
```