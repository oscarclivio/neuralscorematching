# Reproducibility code

## Part 1: Installation

The list of modules in the Python environment used to write this code is available in `environment.yml`. The code is tested with the following setup: Ubuntu 20.04, Python 3.8, CUDA 11.1 (which is a prerequisite to import the environment using `environment.yml`) and the versions of PyTorch, PyTorch-Lightning, numpy, pandas and wandb (see below) as listed in the `environment.yml` file.

To install all Python modules, move to the root folder of the code and run:

``` python
conda env create -f environment.yml --name nsm
```

Activate this environment by running `conda activate nsm`,



## Part 2: Set up Weights & Biases

The code to generate results for neural score matching relies on [Weights & Biases (W&B)](https://wandb.ai).

1. Create a free account if you do not have any account on W&B.

2. Create a project. The project will then have a name in the format `<entity>/<project name>` where `<entity>` is either a username in weights&biases or a team name, depending on where you created your project.

3. Find your API key in **Settings > API keys**.

4. Fill out the JSON file `codes/wandb_config.json` using this information:

   ```yaml
   {
     "entity": "<The <entity> as given in the format <entity>/<project name> in step 2.>",
     "project": "<The <project name> as given in the format <entity>/<project name> in step 2.>",
     "key": "<The API key from step 3.>"
   }
   ```
## Part 3: Run models

1. If not using a GPU, open `codes/train.py `and replace the line 40 :

   ```python
       parser.add_argument('--device', type=str, default="cuda", metavar='N',  # e.g. "cuda", "cpu", ...
   ```

   with

   ```python
       parser.add_argument('--device', type=str, default="cpu", metavar='N',  # e.g. "cuda", "cpu", ...
   ```

2. Go to `codes/`.

3. Generate results from neural score matching by running the following lines. Each of them launches a different sweep on W&B:

   ```bash
   python sweep_config.py --sweep_config sweep_configs/simplematchnet_acic2016_stability.yaml
   python sweep_config.py --sweep_config sweep_configs/simplematchnet_ihdp_stability.yaml
   python sweep_config.py --sweep_config sweep_configs/simplematchnet_news_stability.yaml
   ```

   **Importantly, note the 8-character ID of each sweep that appears when you run these scripts, or find it on the sweeps page in W&B.** Optionally, run `python sweep_config.py --sweep_id <sweep_id>` where `<sweep_id>` is the ID of a given sweep to attach further workers to the sweep and speed up the grid search.

4. Save the metrics from each sweep by running :

   ```bash
   python retrieve_sweep_table.py --sweep_id <sweep_id_1> --table_name acic2016_stability
   python retrieve_sweep_table.py --sweep_id <sweep_id_2> --table_name ihdp_stability
   python retrieve_sweep_table.py --sweep_id <sweep_id_3> --table_name news_stability
   ```

   where each `<sweep_id_1>, <sweep_id_2>, <sweep_id_3>` corresponds to the 3 sweep IDs you noted earlier.

5. To run benchmark models, go to `codes/benchmarks/matchings` and run each Python script in it sequentially:

   ```bash
   python matchings_acic2016_stability.py
   python matchings_ihdp_stability.py
   python matchings_news_stability.py
   ```

   
## Part 4: View results

Go to `notebooks/`: Each notebook corresponds to a different dataset. Run all cells in a given notebook to view results for this dataset.
