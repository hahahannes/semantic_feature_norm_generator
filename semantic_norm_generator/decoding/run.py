import argparse
import pandas as pd
from semantic_norm_generator.decoding.decode import decode_answers, create_rule_dfs_and_save

def run_decode(args):
    answers = pd.read_csv(args.answers, names=['concept', 'answer', 'concept_id', 'run_nr'])
    output_dir = args.output
    lemmatize = args.lemmatize
    number_of_parallel_jobs = args.number_of_parallel_jobs
    keep_duplicates_per_concept = args.keep_duplicates_per_concept
    decoded_answers_df, rule_changes = decode_answers(answers, lemmatize, number_of_parallel_jobs, keep_duplicates_per_concept, output_dir)
    decoded_answers_df.to_csv('%s/decoded_answers.csv' % output_dir, index=False)
    create_rule_dfs_and_save(rule_changes, output_dir)
