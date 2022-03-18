import pandas as pd
import sys


def read_csv(filename, index_col=None):
    if (filename.find(".csv") != -1):
        try:
            df = pd.read_csv(filename, index_col=index_col)
            if (df.shape[1] < 1):
                print(f"Invalid file: {filename}")
                sys.exit(1)
            return df
        except Exception as err:
            print(err)
            sys.exit(1)
    else:
        print(f'Bad file extension: {filename}, please use a ".csv" file')
        sys.exit(1)
