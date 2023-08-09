# Installation
```
git clone https://github.com/hahahannes/semantic_feature_norm_generator
cd semantic_feature_norm_generator
pip install -e .
```

# Usage

## Generation
If you want to generate semantic features for arbitrary words/objects/concepts, you can use the following command. The LLM needs to be primed with examples in order to generate semantic features. You will also need a list of words that you want to generate features for. 

```bash
usage: python -m semantic_norm_generator create [-h] [--output_dir OUTPUT_DIR] [--train_dir TRAIN_DIR] [--retrival_path RETRIVAL_PATH]
                          [--model {gpt3-davinci,gpt-3.5-turbo,gpt-4,gpt-3.5-turbo-0301}] [--model_source {gpt,local,claude}] [--number_runs NUMBER_RUNS] [--number_of_parallel_jobs NUMBER_OF_PARALLEL_JOBS]

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
  --number_of_parallel_jobs NUMBER_OF_PARALLEL_JOBS
                        Number of cores used for generation. Especially useful when APIs are used
```

### Example
The following example uses files from the `doc` directory. Either change the path or run the examples in this directory.
```
python -m semantic_norm_generator create --output_dir=. --train_dir=doc/mcrae_priming --retrieval_path=doc/concepts.csv --model_source=gpt --model=davinci --number_runs=1 --number_of_parallel_jobs=2
```

### Models
The `model` parameter corresponds either to the model name from an API or the name of Huggingface model or the path to weights downloaded from Huggingface. Possible model names from OPENAI can be found here: https://platform.openai.com/docs/models/overview. For example use `davinci` if you want to use the GPT-3 davinci model.  Possible models from Anthropic can be found here: https://docs.anthropic.com/claude/reference/selecting-a-model

### Concepts/Objects/Words
`retrieval_path` specifies the path to the file containing your target concepts/words/objects for which you want to generate semantic features. See the example file in `doc/concepts.csv` for the exptected structure. The question for each words should be in the format of `What are the properties of [WORD]`. Keep singular/plural forms in mind!

### Priming
Note, that the files in the `train_dir` directory need to be named `train_{i}.csv` where `i` specifies the run number. It is recommended to use multiple priming initializations to create more diverse features. Counting starts with `1`. For example, create two files `train_1.csv` and `train_2.csv` if you want to create features with two different primings.
If you do not have priming examples, you can use the McRae priming set in `doc/mcrae_priming`. It consists of 30 different priming runs where each run contains three concept-feature pairs from the McRae feature norm.

### API Keys
If you want to use commercial APIs, e.g. from OpenAI or Anthropic, you need to have an API Key and make it accessible to the tool.
For example by setting the environment variable `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
```
OPENAI_API_KEY=key python -m semantic_norm_generator --help
```

## Decoding
The creation command will create a file containing all the raw text completions from the LLM. To create a feature norm, run `decode` command which will strip off unnecessary pronouns, clean mistakes and transform features. 

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
The following example uses files from the `doc` directory. Either change the path or run the examples in this directory.
```
python -m semantic_norm_generator decode --answers=raw_feature_norm_from_gpt.csv --output=. --parallel
```