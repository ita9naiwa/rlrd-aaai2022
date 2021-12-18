## Implementation of An Efficient Combinatorial Optimization Model Using Learning-to-Rank Distillation

#### This is an offiical implementation of our [work](Paper Link should be here) and baselines and reproducible experiments for the paper.

### Environment Settings
We assume that you have installed Python using [Anaconda](https://docs.anaconda.com/anaconda/install/). It should be possible to use other Python distributions, but we have not tested. Your environment should be equipped with CUDA.

We use Python, PyTorch with CUDA as the backend for our ML implementations.
   * Python >= 3.6
   * PyTorch >= 1.5.0
   * CUDA >= 10.1

We use the packages as listed below(alphabetical order):
   * Cython == 0.29.22
   * [Google OR-tools 9.0.9048](https://developers.google.com/optimization/install/python)


External Libraries:
* PyTorch
* [fast-soft-sort](https://github.com/google-research/fast-soft-sort)
  * Follow instruction guide to install this package.
  * CUDA Extension is required.

#### Installation and how-to-use
---------------

If you are using Conda,
```
conda create --name rlrd python=3.6.9
conda activate rlrd
```
or refer to `virtualenv`.
We divided our implementation into two discrete directories, `mdkp/` for MDKP, and `gfps/` for GFPS.

1. We assume the working directory to be either `mdkp/`. or `gfps/`


2. Install PyTorch with CUDA.
For Conda: `conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch`
For usual Python: `pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
These commands will install appropriate PyTorch with Cuda 10.1 version as the same environment as we test.
Please refer https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/ to see appropriate PyTorch version for your environment.


3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```


4. Dataset Preparation

   a. MDKP
      Dataset made in the fly.

   b. GFPS
      Go to `MDKP/create_dataset.py`, and set the `NUM_PROC_LIST` and `NUM_TASKS_LIST` variables.
      Run `create_dataset.py`, then the dataset will be stored in `../gfpsdata/tr, te, val`.


5. Global model train.
   a. MDKP
      Run `train.py -- num_items 200 --item_dim 20 --w 300 --c 0`. 
      Model will be saved in `../knapsackmodels/globalrl`


      



### Datasets
We use these datasets:
 * SketchFab: https://github.com/EthanRosenthal/rec-a-sketch
 * Epinion: https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm
 * MovieLens-20M: https://grouplens.org/datasets/movielens/20m/
 * Melon: https://arena.kakao.com/c/7/data
We attatch the preprocessed files of the dataset at `code/data/parsed`. You can download and preprocess and create same data using `Data preprocessing.ipynb`  notebook.

### Evaluation Settings:
- We used Early stopping with maximum iteration 300.
  - Validate every 5 epochs (3 for WMF)
  - If the performance does not improve for 3 evaluating in a row, we stop training.
  - For SLIM, we did not conduct early stopping.
  - We used recall@50 as a validation metric. You can change to other metrics.
    - precision/map/ndcg/recall are supported.
- We used batch size 500 for CDAE. We did not see performance differences among different batch sizes.
- For SLIM, we use coordinate descent.
- For CML and DRM, we share data samples for multiple hyperparameters setting up to 8 models. It helps to reduce the expensive number of samplings.

### Frequent Errors
1. `ModuleNotFoundError: No module named 'eval.rec_eval'` error occured.
   * Go to `eval/`, and run the command `python setup.py build_ext --inplace`.


2. `ModuleNotFoundError: No module named 'SLIM'` error occurred.
   * You need to install SLIM package directly from https://github.com/KarypisLab/SLIM.
