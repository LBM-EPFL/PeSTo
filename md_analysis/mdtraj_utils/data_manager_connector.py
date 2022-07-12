import gc
import numpy as np
import pandas as pd
import mdtraj as md
import parmed as pmd

from .trajectory_utils import unwrap_pbc

from data_manager import DataManager


def query_subset(df, pdbid, mdid, rid):
    return df.query('(pdbid == "{}") and (mdid == "{}") and (rid == "{}")'.format(pdbid, mdid, rid))


def load_ref(trajman, df_md, pdbid, mdid):
    # get md info
    md_info_l = query_subset(df_md, pdbid, mdid, 1).to_dict('records')
    # check
    assert len(md_info_l) > 0, f"entry not found for pdbid={pdbid}, mdid={mdid}"
    # find file within database
    db_path = trajman.define_path(md_info_l[0])
    ref_filepath = trajman.find_files(db_path, 'ref.pdb')[0]
    # load structure
    return md.load_pdb(ref_filepath)


def load_full_trajectory(trajman, md_info, mem_opt=True):
    # get current path in database
    db_path = trajman.define_path(md_info)

    # load reference structure
    ref_filepath = trajman.find_files(db_path, 'ref.pdb')[0]
    ref = md.load_pdb(ref_filepath)
    # convert to float32 to reduce memomry usage
    if mem_opt:
        ref.xyz = ref.xyz.astype(np.float32)

    # get all trajectories sorted by pid
    traj_info_dict = {}
    for traj_type in ['nvt', 'npt', 'prod']:
        # fetch info
        traj_info_l = trajman.load_info(db_path, '{}*'.format(traj_type))
        # sort trajectories by pid
        ids_srtd = np.argsort([int(traj_info['pid']) for traj_info in traj_info_l])
        traj_info_dict[traj_type] = [traj_info_l[i] for i in ids_srtd]

    # set order of trajectories
    traj_info_l = []
    # insert NPT/NVT
    if (len(traj_info_dict['nvt']) == 2) and (len(traj_info_dict['npt']) == 2):
        traj_info_l += [traj_info_dict['nvt'][0], traj_info_dict['npt'][0]] + [traj_info_dict['nvt'][1], traj_info_dict['npt'][1]]
    else:
        print("ERROR: NVT and/or NPT missing")
    # insert productions
    if len(traj_info_dict['prod']) > 0:
        traj_info_l += traj_info_dict['prod']
    else:
        print("ERROR: productons missing")

    # load trajectories
    traj_l = []
    for traj_info in traj_info_l:
        # get production info
        traj_type = traj_info['type']
        pid = traj_info['pid']
        # find trajectory in database
        prod_filepath_l = trajman.find_files(db_path, '{}{}.xtc'.format(traj_type,pid))

        # check that file exists
        if len(prod_filepath_l) == 1:
            # get trajectory filepath in the database
            prod_filepath = prod_filepath_l[0]
            # debug print
            # print("Loading {}".format(prod_filepath))
            # load trajectory
            traj_l.append(md.load_xtc(prod_filepath, top=ref.topology))
            # hotfix shift time after npt1 because nvt2 starts at zero
            if '{}{}'.format(traj_info['type'], traj_info['pid']) not in ['nvt1', 'npt1']:
                traj_l[-1].time += 1000.0
            # convert to float32 to reduce memory usage
            if mem_opt:
                traj_l[-1].xyz = traj_l[-1].xyz.astype(np.float32)

        elif len(prod_filepath_l) == 0:
            # no file found
            print("ERROR: {} not found".format('{}{}.xtc'.format(traj_type,pid)))
        else:
            print("WARNING: too many files {}, selecting first one".format(prod_filepath_l))

    # check productions are loaded in the correct order
    for k in range(len(traj_l)-1):
        if traj_l[k].time[-1] > traj_l[k+1].time[0]:
            traj_k_name = '{}{}'.format(traj_info_l[k]['type'], traj_info_l[k]['pid'])
            traj_kp1_name = '{}{}'.format(traj_info_l[k+1]['type'], traj_info_l[k+1]['pid'])
            print("WARNING: time overlap between {} and {}".format(traj_k_name, traj_kp1_name))

    return md.join([ref] + traj_l)


def load_all_ref_pdb(trajman, df_md, pdbid):
    # for each md type
    traj_dict = {}
    for mdid in ['uR', 'uL', 'bR', 'bL', 'C', 'sepB', 'sepU']:
        # find database path
        md_info_l = query_subset(df_md, pdbid, mdid, 1).to_dict('records')
        if len(md_info_l) == 1:
            dbpath = trajman.define_path(md_info_l[0])

            # load pdb
            pdbfile = trajman.find_files(dbpath, "ref.pdb")[0]
            pdb = md.load_pdb(pdbfile)

            # store data
            traj_dict[mdid] = pdb

    return traj_dict


def load_all_data(trajman, df_md, pdbid, mod, rid=1):
    # for each md type
    data_dict = {}
    for mdid in ['uR', 'uL', 'bR', 'bL', 'C', 'sepB', 'sepU']:
        # find database path
        md_info_l = query_subset(df_md, pdbid, mdid, 1).to_dict('records')
        if len(md_info_l) == 1:
            dbpath = trajman.define_path(md_info_l[0])

            # load pdb
            data_l = trajman.load_data(dbpath, mod)

            # if data found
            if len(data_l) == 1:
                # store data
                data_dict[mdid] = data_l[0]
            elif len(data_l) == 0:
                print("WARNING: {} not found for {}".format(mod, mdid))
            else:
                print("WARNING: {} entries of {} found for {}".format(len(data_l), mod, mdid))

    return data_dict


class DataConnector:
    def __init__(self, path, safe=True):
        # create database manager
        self.trajman = DataManager(path, safe=safe)

        # load md info
        md_info_l = self.trajman.load_info(path, 'md')
        self.df_md = pd.DataFrame(md_info_l)

        # prepare buffer
        self.buffer = {}

    def __getattr__(self, name):
        return self.buffer[name]

    def __getitem__(self, name):
        return self.buffer[name]

    def _is_alloc(self, pdbid, mdid, *keys):
        if pdbid not in self.buffer:
            return False
        if mdid not in self.buffer[pdbid]:
            return False
        for key in keys:
            if key not in self.buffer[pdbid][mdid]:
                return False
        # is allocated
        return True

    def _alloc(self, pdbid, mdid):
        # prepare storage
        if pdbid not in self.buffer:
            self.buffer[pdbid] = {}
        if mdid not in self.buffer[pdbid]:
            self.buffer[pdbid][mdid] = {}

    def unload_pdb(self, pdbid):
        # clear pdb data
        if pdbid in self.buffer:
            self.buffer[pdbid] = {}
            gc.collect()

    def unload_md(self, pdbid, mdid):
        if pdbid in self.buffer:
            if mdid in self.buffer[pdbid]:
                # clear md data
                self.buffer[pdbid][mdid] = {}
                gc.collect()

    def load_info(self, pdbid, mdid):
        if not self._is_alloc(pdbid, mdid, 'info'):
            # get md info
            md_info = query_subset(self.df_md, pdbid, mdid, 1).to_dict("records")[0]

            # store data
            self._alloc(pdbid, mdid)
            self.buffer[pdbid][mdid].update({'info': md_info})

    def load_reference(self, pdbid, mdid):
        if not self._is_alloc(pdbid, mdid, 'traj_ref'):
            # load reference
            traj_ref = load_ref(self.trajman, self.df_md, pdbid, mdid)

            # store data
            self._alloc(pdbid, mdid)
            self.buffer[pdbid][mdid].update({'traj_ref': traj_ref})

    def load_trajectory(self, pdbid, mdid):
        if not self._is_alloc(pdbid, mdid, 'traj'):
            # load md info
            self.load_info(pdbid, mdid)

            # get md info
            md_info = self.buffer[pdbid][mdid]['info']

            # load trajectory
            traj = unwrap_pbc(load_full_trajectory(self.trajman, md_info))

            # store data
            self._alloc(pdbid, mdid)
            self.buffer[pdbid][mdid].update({'traj': traj})

    def load_param(self, pdbid, mdid):
        if not self._is_alloc(pdbid, mdid, 'param'):
            # get md info
            md_info = query_subset(self.df_md, pdbid, mdid, 1).to_dict("records")[0]

            # load amberparm for charges and masses
            amberparm = pmd.load_file(md_info['prmtop_filepath'], md_info['inpcrd_filepath'])

            # convert to dataframe
            parmed_df = amberparm.to_dataframe().query("resname != 'WAT' and type != 'K+' and type != 'Cl-'")

            # store data
            self._alloc(pdbid, mdid)
            self.buffer[pdbid][mdid].update({'param': parmed_df})

    def find_data(self, pdbid, mdid, name):
        # get md info
        md_info = query_subset(self.df_md, pdbid, mdid, 1).to_dict("records")[0]

        # define search path
        path = self.trajman.define_path(md_info)

        # return files
        return self.trajman.find_data(path, name)

    def load_data(self, pdbid, mdid, name):
        if not self._is_alloc(pdbid, mdid, name):
            # get md info
            md_info = query_subset(self.df_md, pdbid, mdid, 1).to_dict("records")[0]

            # define search path
            path = self.trajman.define_path(md_info)

            # load data
            self._alloc(pdbid, mdid)
            self.buffer[pdbid][mdid][name] = self.trajman.load_data(path, name)[0]

    def store_data(self, pdbid, mdid, key, data):
        md_info = query_subset(self.df_md, pdbid, mdid, 1).to_dict('records')[0]
        self.trajman.insert_data(md_info, key, data)
