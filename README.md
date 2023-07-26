# Installation
`pip install`

# Usage
## Generation
```
python -m semantic_norm_generator generate --output_dir=. --train_dir=train --retrival_path=priming.csv --model-source=gpt --model=gpt3-davinci
```

## Decoding
```
python -m semantic_norm_generator decode --answers=feature_norm_from_gpt.csv --output=. --parallel
```