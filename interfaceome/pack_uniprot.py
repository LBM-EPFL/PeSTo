import re
import pandas as pd
from glob import glob
from tqdm import tqdm


def main():
    gff_filepaths = glob("uniprot/**/*.gff", recursive=True)

    data = []
    for gff_filepath in tqdm(gff_filepaths):
        with open(gff_filepath, 'r') as fs:
            for line in fs:
                if line[0] != '#':
                    entry = line.strip().split('\t')
                    data.append({
                        'NAME': entry[0],
                        'SOURCE': entry[1],
                        'TYPE': entry[2],
                        'START': entry[3],
                        'END': entry[4],
                        'SCORE': entry[5],
                        'STRAND': entry[6],
                        'FRAME': entry[7],
                        'GROUP': entry[8],
                    })

    # pack into a csv
    df = pd.DataFrame(data)
    df.to_csv("datasets/uniprot_localized_features.csv", index=False)

    # unpack with group data
    data = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        entry_data = {
            'NAME': row['NAME'],
            'SOURCE': row['SOURCE'],
            'TYPE': row['TYPE'],
            'START': row['START'],
            'END': row['END'],
            'SCORE': row['SCORE'],
            'STRAND': row['STRAND'],
            'FRAME': row['FRAME'],
        }

        grp = row['GROUP']
        if len(grp) > 1:
            for entry in grp.split(';'):
                m = re.match(r"(.*)=(.*)", entry)
                entry_data[m[1]] = m[2]

        data.append(entry_data)

    # pack into a csv
    dfs = pd.DataFrame(data)
    dfs.to_csv("datasets/uniprot_localized_features_unwrap.csv", index=False)


if __name__ == '__main__':
    main()
