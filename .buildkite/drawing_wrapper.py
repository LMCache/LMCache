import argparse
import glob
import importlib.util
import os

base_dir = os.path.abspath('../../lmcache-tests/outputs')
process_result_path = os.path.join(base_dir, 'process_result.py')

spec = importlib.util.spec_from_file_location("process_result",
                                              process_result_path)
process_result = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_result)


def process_all_csv_in_directory(directory):
    # Get a list of all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        process_result.process_result_file(csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV files in a directory")
    parser.add_argument("output",
                        type=str,
                        help="The subdirectory to process under lmcache-tests")
    args = parser.parse_args()

    # Specify the directory where the CSV files are located
    process_all_csv_in_directory(args.output)
