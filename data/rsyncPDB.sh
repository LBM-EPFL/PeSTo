#!/bin/sh

# parameters
MIRRORDIR=all_biounits
#MIRRORDIR=all_biounits_cif
LOGFILE=pdb_logs
SERVER=rsync.ebi.ac.uk::pub/databases/rcsb/pdb-remediated
PORT=873
FTPPATH=/data/biounit/PDB/divided/
#FTPPATH=/data/biounit/mmCIF/divided/

# download
rsync -rlpt -v -z --delete --port=$PORT ${SERVER}${FTPPATH} $MIRRORDIR > $LOGFILE 2>/dev/null
