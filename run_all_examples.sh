#! /bin/bash

# RNA folding examples
echo "Start secondary structure training example"
python -m examples.rna_folding_examples.train_example
# python -m examples.rna_folding_examples.biophysical_model_example
echo "Start secondary structure inter family prediction example"
python -m examples.rna_folding_examples.inter_family_prediction_example
echo "Start secondary structure intra family prediction example"
python -m examples.rna_folding_examples.intra_family_prediction_example

# RNA design examples
echo "Start constrained inverse folding example"
python -m examples.rna_design_examples.constrained_design_example
echo "Start inverse folding minimal example"
python -m examples.rna_design_examples.inverse_rna_folding_example_minimal
echo "Start constrained inverse folding with gc content example"
python -m examples.rna_design_examples.inverse_rna_folding_with_gc_content_example
echo "Start riboswitch design example"
python -m examples.rna_design_examples.riboswitch_design_example
echo "Start riboswitch design with properties example"
python -m examples.rna_design_examples.riboswitch_design_with_properties_example
echo "Start riboswitch design torch example"
python -m examples.rna_design_examples.riboswitch_torch_train_example

#General Examples
echo "Start evaluation of all folsing algorithms minimal example"
python -m examples.evaluate_all_folding_baselines_example_minimal
echo "Start f1 score wl comparison example"
# python -m examples.f1score_wl_example
echo "Start visualization example"
python -m examples.visualization_example
echo "Start visualization data length dist example"
python -m examples.visualize_data_length_dist
echo "Start folding prediction visualization example"
python -m examples.visualize_folding_predictions
echo "Start inverse rna folding visualization example"
python -m examples.visualize_inverse_rna_folding_predictions
echo "Start visualization of representations example"
python -m examples.visualize_representations