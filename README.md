# RnaBench
A Comprehensive Library For In Silico RNA Modelling


## Install

### with Anaconda

```
conda env create -f environment.yml
```
Activate the environment with
```
conda activate RnaBench
```
For reproducing all plots of the main paper, you can run
```
./reproduce_all.sh
```
However, this call will take alot of time.
We recommend a cheaper version, that reproduces the plots of the main paper with a limited amount of baselines,
run
```
./reproduce_minimal.sh
```
The install and reproduction scripts were tested on a linux-64 platform.


### with pip
#### Pre-requisites

Ensure you have Python version 3.9 installed. You can check your Python version with:

```bash
python --version
```
f you don't have Python 3.9, you can download it from the [official Python website](https://www.python.org/downloads/).

#### Step 1: Install `RnaBench` via pip

For `infernal`, choose one of the following methods:

   - **Debian package**:
     
     If you are using Debian:

     ```bash
     sudo apt-get install infernal infernal-doc
     ```

   - **Homebrew Science package**:

     If you're using Homebrew:

     ```bash
     brew tap brewsci/bio 
     brew install infernal
     ```

#### Step 2: Install `RnaBench` via pip

##### Option A: From PyPI

```bash
pip install RnaBench
```

##### Option B: Directly from GitHub

To get the latest version directly from the repository:

```bash
pip install git+https://github.com/Rungetf/RnaBench.git
```

## General Usage

We provide a simple API.
All you need to do is to define a function that wraps your model's predictions.
An example script for an RNA Folding model might look like this
```
import RnaBench

benchmark = RnaBench.RnaFoldingBenchmark()

model = YourModel()

def prediction_wrapper(task, *args, **kwargs):
    predicted_pairs = model(task.sequence)
    return predicted_pairs

metrics = benchmark(prediction_wrapper, *args, **kwargs)
print(metrics)
```

## Examples
We provide simple examples with our baselines algorithms.
All examples can be found in the `examples` directory.

You can e.g. run one of our RNA folding examples with
```
python -m examples.rna_folding_examples.<example file without .py suffix>
```


## Data
We provide a bunch of different datasets.
If you are only running the benchmarks, data will be downloaded on the fly as
needed.
However, you can download all datasets by running
```
python -m RnaBench.download
```
Our datasets are pandas dataframes.
If you would like to build your own data with RnaBench, consider at least the following columns in the initial datasets:

- 'sequence': <List[str]>
The sequence as a list of string characters
- 'pairs': <List[Tuple[int, int, int]]>
A list of pairs, provided as triples of (pairing position 1, pairing position 2, level of nesting)
- 'Id': <str/int>
The Id of the sample
- 'has_pk': <bool>
Boolean if the sample contains pseudoknots
- 'has_multiplet': <bool>
Bolean if the sample contains base multiplets
- 'has_nc': <bool>
Boolean if the sample contains non-canonical base pairs
- 'length': <int>
The length of the sequence
- 'gc_content': <float>
The GC-nucleotide ration of the sequence
- 'origin': <str>
Where does this sample come from? Can be used to define datasets, e.g. TS0.

### External Sources
The data processing pipeline for our datasets is also available.
You can remove sequence similarity, blast the training set for homologs
with test samples and query covariance models to further remove redundancy
between training and test data.
However, there are external libraries required
to achieve complete functionality.
To install all external sources, run
```
./install_external_algorithms.sh
```

### Data Processing Pipeline
We provide config examples for our data pipeline at ```RnaBench/lib/data/data_configs```.
You can get a test run of the pipeline with
```
python -m RnaBench.lib.data.build_dataset --config_path RnaBench/lib/data/data_configs/reproduce_inter_family_data_config.yml
```
However, this run might take some minutes.

### Building Covariance Models from Custom Data
```
python -m RnaBench.lib.data.build_cm --df_path <path to dataframe>
```


### Biophysical Model data
Our pipeline includes a flag for sampling sequences from Rfam covariance models.
The ```Rfam.cm``` and its preparation are part download script.
```
python -m RnaBench.download
```
We provide an example config file for the Biophysical model pipeline at ```RnaBench/lib/data/data_configs/biophysical_example_config.yml```.

### 3D data
We use the data from ```RNAsolo``` for our 3D RNA data pipeline and provide a parser for the provided ```mmCIF``` structure files using ```Biopython```.
However, RNAsolo is updated very regularly to follow a new BGSU version.

We *note*: Whenever RNAsolo updates to a new BGSU version, our download of 3D data might not work out-of-the-box.
You can change the BGSU version for the download script in the ```RnaBench/download.py``` script.
you can look up the BGSU version of RNAsolo in the bottom left corner here:
```
https://rnasolo.cs.put.poznan.pl/archive
```
However, we also provide PDB's download script at ```RnaBench/lib/data/threedee ``` that you can also use for downloading specific PDB IDs.
Afterwards, you can run your model with RnaBench's dataset for 3D data as well and a torch dataloader.
Moreover, we provide the representatives of the equivalence-classes for a resolution threshold at 1.5 angstrom in the data directory.
An example script for the usage of our 3D data pipeline, including RMSD computation looks as follows
```
from torch.utils.data import DataLoader
from Bio.SVDSuperimposer import SVDSuperimposer

from RnaBench.lib.data.threedee.data import Rna3dDataset

mmcif_dir = "<path to directory containing mmCIF files>"

rna_3d_dataset = Rna3dDataset(mmcif_dir, device='cpu')
sup = SVDSuperimposer()
data_iterator = DataLoader(rna_3d_dataset, batch_size=64)

for i_batch, sampled_batch in enumerate(data_iterator):
    for b, length in enumerate(sampled_batch["sequence_length"].detach().cpu().numpy()):
        rna_sequence = [rna_3d_dataset.nucleotide_itos[i] for i in sampled_batch['sequence'][b, :length].detach().cpu().numpy()]
    for b, length in enumerate(sampled_batch["length"].detach().cpu().numpy()):
        true_x = sampled_batch['x_coordinate'][b, :length].detach().cpu().numpy()
        true_y = sampled_batch['y_coordinate'][b, :length].detach().cpu().numpy()
        true_z = sampled_batch['z_coordinate'][b, :length].detach().cpu().numpy()

        # do some prediction to get x, y, z coordinates for each atom
        x, y, z = model(rna_sequence)

        true_coords = np.stack([true_x, true_y, true_z], axis=1)
        pred_coords = np.stack([x, y, z], axis=1)

        sup.set(true_coords, pred_coords)
        sup.run()
        rmsd = sup.get_rms()
        print('RMSD:', np.round(rmsd, 4))
```

## Baselines
Our baselines are either implemented within RnaBench, or are part of the installation of external algorithms.
