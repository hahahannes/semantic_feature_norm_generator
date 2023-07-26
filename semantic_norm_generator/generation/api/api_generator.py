import os
import sys
sys.path.append('..')

import pandas as pd 
import argparse
import multiprocessing as mp 

from semantic_norm_generator.generation.generation_jobs import yield_generation_jobs

class APIGenerator():
    def __init__(
        self,
        output_dir,
        model,
        train_dir,
        retrieval_path
    ):

    def run(self):
        output_dir = self.output_dir
        model = self.model
        data_path = f'{output_dir}/encoded_answers_openai.csv'
        train_dir = self..train_dir 
        retrival_path = self.retrival_path

        #must use Manager queue here, or will not work
        manager = mp.Manager()
        output_queue = manager.Queue()    
        job_queue  = manager.Queue()  
        n_jobs = 3 #mp.cpu_count()
        pool = mp.Pool(n_jobs)

        #put ouput listener to work first
        print('Start Output Writer')
        watcher = pool.apply_async(listener, (output_queue, data_path))

        #fire off workers
        print('Start Workers')
        jobs = []
        for i in range(n_jobs):
            job = pool.apply_async(worker, (i, job_queue, output_queue, model))
            jobs.append(job)

        # create jobs
        for job in yield_generation_jobs(data_path, train_dir, retrival_path):
            job_queue.put(job)

        for _ in jobs:
            job_queue.put('kill')

        print(f'Number of jobs: {job_queue.qsize()}')
            
        print('Wait for workers to finish')
        for job in jobs:
            job.get()

        print('Workers are done!')
        output_queue.put('kill')
        watcher.get()
        print('Output Writer is done!')

        pool.close()
        pool.join()


    def escape_answer(self, text):
        return text.replace('\n', '').replace('"', '""')

    def worker(self, i, job_queue, output_queue, model):
        print(f'start worker {i}')
        while 1:
            job = job_queue.get()
            if job == 'kill':
                break

            priming = job['priming']
            question = job['question']
            answer = make_request(priming, model, question)
            answer = escape_answer(answer)
            job['answer'] = answer
            output_queue.put(job)
        return True 

    def listener(self, output_queue, out_path):
        '''listens for messages on the q, writes to file. '''
        with open(out_path, 'a+') as f:
            while 1:
                result = output_queue.get()
                if result == 'kill':
                    break

                cocnept = result['concept']
                answer = result['answer']
                cocnept_id = result['concept_id']
                run_nr = result['run_nr']
                text = f'{cocnept},"{answer}",{cocnept_id},{run_nr}'
                f.write(text + '\n')
                f.flush()

        
        return True