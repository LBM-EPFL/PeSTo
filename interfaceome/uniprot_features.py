import os
import json
from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_features(xml_filepath):
    soup = BeautifulSoup(open(xml_filepath,'r'), 'lxml')
    features = []
    for ftr in soup.find_all("feature"):
        # extract feature info
        ftype = ftr['type']
        if ftr.has_attr('description'):
            desc = ftr['description']
        else:
            desc = None

        # extract position
        pos_tag = ftr.find('position')
        if pos_tag is not None:
            pos = int(pos_tag['position'])
        else:
            begin_tag = ftr.find('begin')
            end_tag = ftr.find('end')
            if begin_tag.has_attr('position') and end_tag.has_attr('position'):
                p0 = int(ftr.find('begin')['position'])
                p1 = int(ftr.find('end')['position'])
                pos = (p0,p1)
            else:
                continue

        features.append({'ftype':ftype, 'pos': pos, 'desc': desc})

    return features


def main():
    # xml_filepaths = glob("uniprot/HUMAN/**/*.xml", recursive=True)
    xml_filepaths = glob("uniprot/**/*.xml", recursive=True)

    features_dict = {}
    for xml_filepath in tqdm(xml_filepaths):
        uniprot = os.path.basename(xml_filepath).replace('.xml', '')
        features_dict[uniprot] = extract_features(xml_filepath)

    json.dump(features_dict, open("datasets/uniprot_features.json", 'w'))


if __name__ == '__main__':
    main()
