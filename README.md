# Installation
`pip install -e .`

# Usage
## Generation
```bash
usage: python -m semantic_norm_generator create [-h] [--output_dir OUTPUT_DIR] [--train_dir TRAIN_DIR] [--retrival_path RETRIVAL_PATH]
                          [--model {gpt3-davinci,gpt-3.5-turbo,gpt-4,gpt-3.5-turbo-0301}] [--model_source {gpt,local,claude}] [--number_runs NUMBER_RUNS]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Path to directory where the file containing the raw completions is saved
  --train_dir TRAIN_DIR
                        Path to the directory containing the priming examples. Each file in the directory corresponds to a specific priming
  --retrival_path RETRIVAL_PATH
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
python -m semantic_norm_generator generate --output_dir=. --train_dir=train --retrival_path=priming.csv --model-source=gpt --model=gpt3-davinci
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