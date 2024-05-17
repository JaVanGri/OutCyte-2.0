
<img src="logo.svg" alt="OutCyte 2.0 Logo" style="width:100%; text-align:center;">

## What is OutCyte 2.0?
**OutCyte 2.0** is an advanced tool designed for predicting various cellular localization attributes including signal peptides, transmembrane regions, unconventional protein secretion (UPS), and intracellular proteins, based on the amino acid sequence of proteins. It is an enhancement over the previous tool developed by Linlin Zhao, which can be found at [Linlin Zhao's OutCyte](https://github.com/linlinzhao/outcyte). This version significantly improves the prediction accuracy for UPS proteins.

If you prefer there also exists a GUI version of this tool  [OutCyte-2.0-Web](https://github.com/JaVanGri/OutCyte-2.0-Web).
## How to use it?
To effectively utilize OutCyte 2.0, please follow these steps:

1. **Install the dependencies**:
   - Ensure you have Conda installed on your system.
   - All necessary dependencies are listed in `environment.yml`. Install them by running the following command in the terminal:
   ```
   conda env create -f environment.yml
   ```

2. **Activate the Conda environment**:
   ```
   conda activate oc2
   ```

3. **Prepare your FASTA file**:
   - Ensure your FASTA file is formatted correctly with sequences you wish to analyze.

4. **Run the application**:
   - Execute the application by running the command below, specifying the path to your FASTA file, the desired mode of operation, and the computation device (`cpu` or `cuda`):
   ```
   python execute.py /path/to/your/fasta/file.fasta --mode standard_v2 --device cpu
   ```
   - Available modes are:
        - standard_v2`: Predicts Signal Peptide, Transmembrane, UPS (new model), and Intracellular
        - `standard`: Uses the old UPS model; predicts Signal Peptide, Transmembrane, and Intracellular.
        - `sp`: Targets Signal Peptide, Transmembrane, and Intracellular.
        - `ups`:  Focuses on UPS (old model) and Intracellular.
        - `ups_v2`: Uses the updated UPS model to differ between UPS and Intracellular.

5. **Access the results**:
   - The result file will be created in the same directory as your input file, named with the mode of operation and additional "_RESULT.csv" suffix.

Please note: Adjust the `--max_sequence_length` and `--min_sequence_length` parameters as needed to tailor the analysis to your sequences.


## Retraining

If you want to retrain the model on a new dataset, you can do so easily. The training data should be in a CSV file with columns **Entry**, **Sequence**, and **Label**.

You start the training with:

```
   python retraining.py training_data.csv
```
