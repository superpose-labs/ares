First, download and ingest kaist dataset:
- python scripts/kaist/ingest_kaist.py

Then build QUBO coefficients:
- python scripts/curation/build_window_qubo_instance.py

Check the output in 'ares/data/curation/kaist_window_instance.pkl'.

To understand the coefficients created, see 'QUBO_KAIST_PLAN.md'.


Run 'run_batch_experiments.py' to solve and select subset of data.
- python scripts/curation/run_batch_experiments.py


Run 'python scripts/curation/create_curated_rlds_datasets.py' to read the output from experiment and create subset of data


Summary
```
python scripts/kaist/ingest_kaist.py
python scripts/curation/build_qubo_instance.py 
python scripts/curation/run_batch_experiments.py
python scripts/curation/create_curated_rlds_datasets.py
```
