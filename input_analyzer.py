import argparse
import pandas as pd

if __name__ == "__main__":

    # Argumentos del main
    parser = argparse.ArgumentParser(description='Parser for PyTextRank parameters')
    parser.add_argument('--filename', metavar='path', required=True, help='the name of the input file')

    args = parser.parse_args()

    print("Reading Text...")
    dataset = pd.read_json('./dataset/' + args.filename + '.json')