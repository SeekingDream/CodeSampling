# CodeSampling

1. **TODO:** finish readme 
2. **TODO:** load_perturbed_prompt in `utils.py` should return `MyDataset`, instead of just prompt
3. **TODO:** modify generate_code.py to add `stop` tokens
4. **TODO:** implement the bash script, including a pipeline for one model and one dataset.
   A demo could be 
   1. ``python generaye_code.py --model_name --data_name --hyper_name --n``  # Generate code, save code to XXX
   2. ``python execute_code.py --code_dir``                                  # Execute code, save results to XXX
   3. ``python evaluate_sampling.py --sampling_alg``                         # Sampling Code, save sampling results to XXX
   4. ``python post.py``                                                     # Post Process, save results to csv files.

5. **TODO:** where is the reranking algorothm implementation?
6. **TODO:** try running a demo and show the results dir. An demo could be 

```
results/
├── generated_code/
│   └── hyper_param/
│       └── model_name/
│           └── data_name/
│               └── try_id/
│                   ├── problem_1.py
│                   ├── problem_2.py
│                   └── ...
├── correctness_res/
│   └── hyper_param/
│       └── model_name+data_name/
│           └── is_correct.json
└── sampling_res/
    └── ...
```


