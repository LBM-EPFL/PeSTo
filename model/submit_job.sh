#!/bin/sh
python=$HOME/miniconda3/envs/pesto/bin/python

# get run info
output_dir=$($python -B -c "import config; print(config.config_runtime['output_dir'])")
run_name=$($python -B -c "import config; print(config.config_runtime['run_name'])")

# clear existing directory
if [ -d "$output_dir/$run_name" ]; then
    echo "Removing previous session: $output_dir/$run_name"
    rm -rf "$output_dir/$run_name"
fi

# create save directories
mkdir -p "$output_dir/$run_name"
mkdir -p "$output_dir/$run_name/tb"

# copy source files
cp *.py "$output_dir/$run_name"
cp -L -r src "$output_dir/$run_name"

# link data
ln -rsf datasets "$output_dir/$run_name"

# copy job file
cp train.job "$output_dir/$run_name"

# go to working directory
cd "$output_dir/$run_name"

# debug print
echo "Working directory: $(pwd)"

# queue job
sbatch -J "$1" train.job
