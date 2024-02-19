#! /bin/bash

echo "Reproduce Main paper plots"

# echo "### Build environment"

# conda env create -f environment.yml
# conda activate RnaBench

echo "### Download secondary structure data"

# python -m RnaBench.download

echo "### Make external algorithms"

# ./install_external_algorithms.sh

echo "### Run all folding predictions"

python -m examples.evaluate_all_folding_baselines_example_minimal

echo "### Visualize results"

python -m examples.visualize_folding_predictions

echo "### Run inverse RNA folding benchmark"

python -m examples.rna_design_examples.inverse_rna_folding_example_minimal

echo "### Plot Inverse RNA Folding results"

python -m examples.visualize_inverse_rna_folding_predictions

echo "### Visualize 5SrRNA to compare Wesifeiler-Lehman, F1-scores and MCC"

python -m examples.f1score_wl_example

echo "### Plot datasets"

python -m RnaBench.download
python -m examples.visualize_data_length_dist

echo "### Plot example representations"

python -m examples.visualize_representations