import sys
from datetime import datetime
from src.data_encoding import categ_to_resnames


config_data = {
    # 'dataset_filepath': "datasets/contacts_rr5A_64nn_8192.h5",
    'dataset_filepath': "/tmp/"+sys.argv[-1]+"/contacts_rr5A_64nn_8192.h5",
    'train_selection_filepath': "datasets/subunits_train_set.txt",
    'test_selection_filepath': "datasets/subunits_test_set.txt",
    'max_ba': 1,
    'max_size': 1024*8,
    'min_num_res': 48,
    'l_types': categ_to_resnames['protein'],
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna']+categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ],
    # 'r_types': [[c] for c in categ_to_resnames['protein']],
}

config_model = {
    "em": {'N0': 123, 'N1': 32},
    "sum": [
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
    ],
    "spl": {'N0': 32, 'N1': 32, 'Nh': 4},
    "dm": {'N0': 32, 'N1': 32, 'N2': 5}
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'i_v3_0'+tag,
    'output_dir': 'save',
    'reload': True,
    'device': 'cuda',
    'num_epochs': 100,
    'batch_size': 1,
    'log_step': 1024,
    'eval_step': 1024*8,
    'eval_size': 1024,
    'learning_rate': 1e-5,
    'pos_weight_factor': 0.5,
    'comment': "",
}
