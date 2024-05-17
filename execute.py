import argparse
import logging
import os
import sys

from bin.models.outcyte_sp import run_sp
from bin.models.outcyte_ups import run_ups
from bin.models.outcyte_ups_v2 import predict_ups, read_fasta

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def run_standard_mode(file_data):
    results = run_sp(file_data['Entry'], file_data['Sequence'])
    ups_results = run_ups(file_data['Entry'], file_data['Sequence'])
    results['ups'] = (1 - results['transmembrane'] - results['signal_peptide']) * ups_results['ups']
    results['intracellular'] = 1 - results['transmembrane'] - results['signal_peptide'] - results['ups']
    return results


def run_standard_v2_mode(self, file_data, device):
    results = run_sp(file_data['Entry'], file_data['Sequence'])
    ups_results = predict_ups(file_data['Entry'], file_data['Sequence'], device=device)
    results['ups'] = (1 - results['transmembrane'] - results['signal_peptide']) * ups_results['ups']
    results['intracellular'] = 1 - results['transmembrane'] - results['signal_peptide'] - results['ups']
    return results


def dict_to_string(dict):
    result = ""
    for key in dict.keys():
        result = result + key + ": " + str(dict[key]) + " | "
    return result


def get_predictions(mode, file_data, device):
    if mode == 'standard':
        return run_standard_mode(file_data)
    elif mode == 'sp':
        return run_sp(file_data['Entry'], file_data['Sequence'])
    elif mode == 'ups':
        return run_ups(file_data['Entry'], file_data['Sequence'])
    elif mode == 'standardv2':
        return run_standard_v2_mode(file_data)
    else:
        return predict_ups(file_data['Entry'], file_data['Sequence'], device=device)


def find_max_key(result, keys):
    keys = list(set(list(result.keys())).intersection(set(keys)))
    return max(result[keys].to_dict(), key=result.get)


def count_classes(results):
    prediction_counts = {'transmembrane': 0, 'signal_peptide': 0, 'ups': 0, 'intracellular': 0}
    for i in range(len(results)):
        result = results.iloc[i]
        max_key = find_max_key(result, ['transmembrane', 'signal_peptide', 'ups', 'intracellular'])
        prediction_counts[max_key] += 1
    return prediction_counts


def process_input(file_path, max_sequence_length=2700, min_sequence_length=20, allowed_character=None):
    if allowed_character is None:
        allowed_character = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    try:
        file_data = read_fasta(file_path)
    except FileNotFoundError:
        return None, f"File not found."
    file_data = file_data[file_data['Sequence'].str.len() < max_sequence_length]
    file_data.index = range(len(file_data))
    file_data = file_data[file_data['Sequence'].str.len() > min_sequence_length]
    file_data.index = range(len(file_data))

    file_data['Sequence'] = file_data['Sequence'].str.upper()
    for entry, sequence in file_data[['Entry', 'Sequence']].values:
        if not all(char in allowed_character for char in sequence):
            return None, f"Invalid characters in sequence of {entry}!"

    return file_data, None


def main():
    # Clear the console screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Predict secretion pathways based on FASTA file input.")
    parser.add_argument('filepath', type=str, help="Path to the FASTA file")
    parser.add_argument('--max_sequence_length', type=int, default=2700, help="Maximum sequence length")
    parser.add_argument('--min_sequence_length', type=int, default=20, help="Minimum sequence length")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Compute device to use ('cpu' or 'cuda')")
    parser.add_argument('--mode', type=str, default='standard_v2',
                        choices=['sp', 'ups', 'ups_v2', 'standard', 'standard_v2'],
                        help="Select the prediction mode. Modes include 'sp', 'ups', 'ups_v2', 'standard', "
                             "and 'standard_v2'.")

    args = parser.parse_args()

    # Process the input file
    file_data, error_message = process_input(file_path=args.filepath,
                                             max_sequence_length=args.max_sequence_length,
                                             min_sequence_length=args.min_sequence_length)

    if error_message:
        logging.error(error_message)
        sys.exit(1)

    # Perform predictions
    logging.info(f"Calculating predictions for {len(file_data)} proteins.")
    results = get_predictions(args.mode, file_data, args.device)
    logging.info("Predictions completed successfully.")

    # Output results
    result_path = args.filepath.replace('.', '_') + f'_{args.mode}_RESULT.csv'
    results.to_csv(result_path)
    logging.info(f"Results saved at {result_path}")


if __name__ == "__main__":
    main()
