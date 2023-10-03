import pandas as pd 

def yield_generation_jobs(raw_feature_path, train_dir, retrival_path, number_runs):
    retrieval_df = pd.read_csv(retrival_path)

    try:
        current_answers_saved = pd.read_csv(raw_feature_path, names=['concept', 'answer', 'concept_id', 'run_nr'])
    except IOError:
        current_answers_saved = pd.DataFrame({'concept_id': [], 'run_nr': [], 'answer': []})

    print("Peek at already saved completions:")
    print(current_answers_saved.head())

    for run_nr in list(range(1, number_runs+1)):  
        train_file_name = f"train_{str(run_nr)}.csv"
        print(f'Check {train_file_name}')
        train_df = pd.read_csv('%s/%s' % (train_dir, train_file_name))

        for row in retrieval_df.itertuples():
            concept_id = row.id
            concept_run_already_sampled = (((current_answers_saved.concept_id == concept_id) & (current_answers_saved.run_nr == run_nr)).any())
            concept_occurs_in_priming = concept_id in list(train_df[:3]['concept'])

            if concept_run_already_sampled or concept_occurs_in_priming:
                print(f'Skip {concept_id} - {run_nr}')
                continue
        
            question = row.question
                
            priming = []
            for priming_example in train_df.itertuples():
                priming.append([priming_example.question, priming_example.answer])

            job = {
                'priming': priming,
                'run_nr': run_nr,
                'concept': row.concept,
                'concept_id': concept_id,
                'question': question
            }
            yield job

def yield_generation_jobs_in_batches(raw_feature_path, train_dir, retrival_path, number_runs):
    batches = []
    for job in yield_generation_jobs(raw_feature_path, train_dir, retrival_path, number_runs):
        batches.append(job)

        if len(batches) == 16:
            yield batches
            batches = []