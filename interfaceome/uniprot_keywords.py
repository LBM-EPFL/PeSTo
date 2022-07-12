import os
import json
from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_keywords_locations(xml_filepath):
    soup = BeautifulSoup(open(xml_filepath, 'r'), 'lxml')

    sc_kws = soup.find_all("keyword")

    keywords = []
    for sc_kw in sc_kws:
        keywords.append(sc_kw.text)

    return keywords


def main():
    xml_filepaths = glob("uniprot/HUMAN/**/*.xml", recursive=True)

    kws_dict = {}
    for xml_filepath in tqdm(xml_filepaths):
        uniprot = os.path.basename(xml_filepath).replace('.xml', '')
        kws_dict[uniprot] = extract_keywords_locations(xml_filepath)

    json.dump(kws_dict, open("datasets/uniprot_keywords.json", 'w'))


if __name__ == '__main__':
    main()
