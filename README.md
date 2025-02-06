# CktGNN
CktGNN is a two-level GNN model with a pre-designed subgraph basis for the analog circuit (DAG) encoding. CktGNN simultaneously optimizes circuit topology and device features, achieving state-of-art performance in analog circuit optimization. <br/>For more information, please check our ICLR 2023 paper: 'CktGNN: Circuit Graph Neural Network for Electronic Design Automation' (https://openreview.net/forum?id=NE2911Kq1sp).

## Environment Setup

### Prerequisites
- Python 3.7+ (Python 3.9 recommended, especially for Apple Silicon/ARM64 machines)
- pip (Python package installer)
- Docker (optional, for containerized setup)

> **Important Note for Apple Silicon (M1/M2) Users**: 
> If you're using an Apple Silicon (ARM64) Mac, we strongly recommend using the Conda installation method below
> for better compatibility with scientific packages and PyTorch Geometric.

### Option 1: Docker Installation (Recommended for reproducibility)

1. Install Docker for your platform from [docker.com](https://www.docker.com/products/docker-desktop)

2. Build and run with docker-compose:
```bash
# Build the image
docker-compose build

# Verify the environment
docker-compose run --rm cktgnn python env_validation.py

# Run training
docker-compose run --rm cktgnn python main.py --epochs 300 --save-appendix _cktgnn --model CktGNN --hs 301
```

The Docker setup:
- Uses Python 3.9 with all dependencies pre-configured
- Mounts local OCB and results directories
- Caches pip packages for faster rebuilds
- Includes environment validation to ensure everything is working
- Handles platform-specific dependencies automatically

#### Apple Silicon (M1/M2) Optimizations
The Docker setup includes specific optimizations for Apple Silicon:
- Uses native ARM64 containers
- Enables Metal Performance Shaders
- Optimizes BLAS operations for Apple Silicon
- Configures thread allocation for M-series chips

Performance on M2 Max:
- ~1.7 iterations/second with optimized Docker setup
- Efficient multi-core utilization
- Native-like performance with containerization

Key optimizations:
- ARM64-native Python base image
- OpenBLAS and BLAS optimizations
- Thread and memory tuning
- Proper worker process allocation

### Option 2: Standard Installation (pip)

> **Note**: This method may not work on Apple Silicon Macs or if you need specific
> version compatibility. In those cases, use Option 2 (Conda Installation) instead.

1. Create a virtual environment:
```bash
# For most systems:
python3.9 -m venv venv

# If you don't have Python 3.9 installed on macOS:
brew install python@3.9
```

2. Activate the virtual environment:

On macOS/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
# Note: This may not work on all platforms. If you encounter issues,
# use Option 2 (Conda Installation) below instead.

# Install packages in the correct order
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.3.1
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1
pip install -r requirements.txt
```

### Option 2: Conda Installation (Recommended for Apple Silicon/ARM64)

1. Install Miniconda if not already installed:
```bash
# On macOS
brew install --cask miniconda

# Initialize conda (choose based on your shell)
conda init zsh  # for zsh (default on newer Macs)
# OR
conda init bash # for bash
```

2. Close and reopen your terminal, then create and activate a new conda environment:
```bash
conda create -n cktgnn python=3.9
conda activate cktgnn
```

3. Install PyTorch and PyTorch Geometric:
```bash
# Add conda-forge channel (order matters)
conda config --add channels conda-forge

# Install PyTorch 1.13.1 (stable version for Apple Silicon)
pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install PyTorch Geometric and extensions via pip
pip install torch-geometric==2.3.1
# Install PyG extensions (order matters)
pip install torch-scatter==2.1.1
pip install torch-sparse==0.6.17
pip install torch-cluster==1.6.1

# Install other dependencies
conda install -c conda-forge networkx python-igraph gensim tqdm pillow pandas scikit-learn matplotlib
```

### Platform-Specific Notes

#### Apple Silicon (M1/M2) Macs
- Use Option 2 (Conda Installation) above - it's specifically tested for Apple Silicon
- The versions specified are known to work together on ARM64
- If you see a warning about torch_spline_conv, it's safe to ignore - this is an optional dependency
- Make sure to install packages in the order shown above

#### CUDA Support
This installation is CPU-only. For CUDA support on other platforms, please visit https://pytorch.org/get-started/locally/

### Verifying the Installation

You can verify your installation by running:
```bash
python env_validation.py
```

This will:
- Check all required packages and their versions
- Verify PyTorch Geometric and its extensions are working
- Test basic GNN operations
- Show detailed environment information

* The experiment codes are basically constructed upon [D-VAE](https://github.com/muhanzhang/D-VAE/).

## OCB: Open Circuit Benchmark

* OCB is the first open benchmark dataset for analog circuits (i.e. operational amplifiers (Op-Amps)), equipped with circuit generation code, evaluation code, and transformation code that converts simulation results to different graph datasets (i.e. igraph, ptgraph, tensor.) Currently, OCB collects two datasets, Ckt-Bench-101 and Ckt-Bench-301 for the general-prpose graph learning tasks on circuits. 

* Ckt-Bench-101 (directory: `/OCB/CktBench101`): Ckt-Bench-101 is generated based on the dataset used in our ICLR 2022 paper. Ckt-Bench-101 contains 10000 different circuits, and it eliminates invalid circuits in the datasets used in our ICLR 2023 paper and replace them with new valid simulations. 

* Ckt-Bench-301 (directory: `/OCB/CktBench301`): Ckt-Bench-301 contains 50000 circuits. Circuits in Ckt-Bench-301 and in Ckt-Bench-101 have different device features and circuit topologies. This benchmark dataset is proposed to perform the Bayesian optimization, as it might be hard to implement simulation code on the circuit simulators without relevant expertise.

* Source code (directory: `/OCB/src`): The source codes enable users to construct their own analog circuit datasets of arbitrary size. The directory `/OCB/src` provides simulation codes for circuit simulators. [/OCB/src/circuit_generation.py]( /OCB/src/circuit_generation.py) generates circuits and writes them in .txt file. [/OCB/src/utils_src.py](/OCB/src/utils_src.py) includes functions (train_test_generator_topo_simple) that convert the txt circuits and relevant simulation results to igraph data.

* Tutorial (directory: `/OCB`): The tutorial [/OCB/Tutorial.pdf](/OCB/Tutorial.pdf) provides guidance of understanding the source code and implementing the performance simulation on circuit simulators.  

## Experiments on Ckt-Bench-101

* Run variatioal auto-encoders on Ckt-Bench-101 to train different circuit/DAG encoders (CktGNN, PACE, DVAE, DAGNN...), and test the decoding ability of the decoder in the VAE (proportion of valid DAGs, valid circuits, novel DAGs generated from the latent encoding space of the circuit encoder, proportion of accurately reconstructed DAGs.) 

`python main.py --epochs 300 --save-appendix _cktgnn --model CktGNN --hs 301`

User can select different models (e.g. DVAE, DAGNN ..) and uses the corresponding save appendix (e.g. `--save-appendix _dvae`, `--save-appendix _dagnn` ) to store the results.

* Run SGP regression to test whether the circuit encoder can generate a smooth latent space w.r.t. circuit properties. There are many circuit properties: FoM, Gain, Bw, Pm, while FoM is the most critical one. In circuit optimization, FoM characterizes the circuit quality and the objective is to maximize circuit's FoM.

`