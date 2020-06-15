#!/bin/bash

echo "Installing local python env..."

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cd /home/e2crawfo/spair_video/silot
pip install --no-index -r beluga_requirements.txt

cd /home/e2crawfo/spair_video/clify
pip install -e .

cd /home/e2crawfo/spair_video/auto_yolo/
pip install -e .

cd /home/e2crawfo/spair_video/dps
pip install -e .

cd /home/e2crawfo/spair_video/sqair
pip install -e .

cd /home/e2crawfo/spair_video/silot
pip install -e .
