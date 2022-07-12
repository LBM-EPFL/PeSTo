import os
import json
from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_subcellular_locations(xml_filepath):
    soup = BeautifulSoup(open(xml_filepath, 'r'), 'lxml')

    sc_locs = soup.find_all("subcellularlocation")

    locations = []
    for sc_loc in sc_locs:
        locations.extend([c.text for c in sc_loc.find_all("location")])

    return locations


def main():
    xml_filepaths = glob("uniprot/HUMAN/**/*.xml", recursive=True)

    locs_dict = {}
    for xml_filepath in tqdm(xml_filepaths):
        uniprot = os.path.basename(xml_filepath).replace('.xml', '')
        locs_dict[uniprot] = extract_subcellular_locations(xml_filepath)

    json.dump(locs_dict, open("datasets/subcellularlocation.json", 'w'))


if __name__ == '__main__':
    main()
