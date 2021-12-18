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
      - Dataset made in the fly.

   b. GFPS
      - Go to `MDKP/create_dataset.py`, and set the `NUM_PROC_LIST` and `NUM_TASKS_LIST` variables.
      - Run `create_dataset.py`, then the dataset will be stored in `../gfpsdata/tr, te, val`.


5. Global model train.
   a. MDKP
      - Run `train.py --num_items 200 --item_dim 20 --w 300 --c 0`. Add `--use_cuda` if possible.
      - Model will be saved in `../mdkpmodels/globalrl`

   b. GFPS
      - Run `train.py --num_tasks 32 --num_procs 4`, or according to your dataset created.
      - Model will be saved in `../gfpsmodels/globalrl`

6. Model train.
   Each model will be trained by tuning the global model.
   a. MDKP
      - Run `localtrain.py --num_items 50 --item_dim 3 --w 200 --c 0`. Add `--use_cuda` if possible.
      - Model will be saved in `../mdkpmodels/localrl`

   b. GFPS
      - Run `localtrain.py --num_tasks 32 --num_procs 4 --range_r 3.00 --range_l 3.00`. Please set `range_r = range_l`.
      - Model will be saved in `../gfpsmodels/localrl`

7. Distillation
   a. MDKP
      - Run `distillation.py --num_items 50 --item_dim 3 --w 200 --c 0`. 
      - Each parameters num_items, item_dim, w, and c must be trained on localRL by the previous step 6.
      - Model will be saved in `../mdkpmodels/distillation`
   
   b. GFPS
      - Run `distillation.py --num_tasks 32 --num_procs 4 --range_r 3.00 --range_l 3.00`. Please set `range_r = range_l`.
      - Each parameters num_tasks, num_procs and range must be trained on localRL by the previous step 6.



### Evaluation:
a. MDKP
   - Run `test.py --num_items 50 --item_dim 3 --w 200 --c 0`. 
   - You can see ratio of performance and inference time.
   
b. GFPS
   - Run `test.py --num_tasks 32 --num_procs 4 --range_r 3.00 --range_l 3.00`. Please set `range_r = range_l`. 
   - You can see ratio of performance and inference time.
   
