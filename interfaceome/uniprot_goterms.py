import os
import json
from tqdm import tqdm
from glob import glob
from bs4 import BeautifulSoup


def extract_goterms(xml_filepath):
    soup = BeautifulSoup(open(xml_filepath, 'r'), 'lxml')

    goterms = {}
    for goterm_ext in soup.find_all("dbreference", {"type":"GO"}):
        for goterm_tag in goterm_ext.find_all("property", {"type":"term"}):
            gt = goterm_tag['value']
            tag = gt.split(':')[0]
            term = gt.split(':')[-1]
            if tag in goterms:
                goterms[tag].append(term)
            else:
                goterms[tag] = [term]

    return goterms


def main():
    xml_filepaths = glob("uniprot/HUMAN/**/*.xml", recursive=True)

    goterms = {}
    for xml_filepath in tqdm(xml_filepaths):
        uniprot = os.path.basename(xml_filepath).replace('.xml', '')
        goterms[uniprot] = extract_goterms(xml_filepath)

    json.dump(goterms, open("datasets/goterms.json", 'w'))


if __name__ == '__main__':
    main()
