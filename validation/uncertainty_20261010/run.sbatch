#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J blrfit-uncertainty
#SBATCH -o %x-%j.out
# The single-spectrum uncertainty study of blrfit, repeated with the frozen wheel that prepare.py put in
# installed/ next to this file. Submit from the bundle directory on Perlmutter:
#     cd /pscratch/sd/e/eka/_2BINARIES/blrfit_uncertainty_20261010 && sbatch run.sbatch
# The results go to $PSCRATCH/SINGLE_UNCERTAINTY_<job id>/ and the compact return archive next to it
# (RETURN_FILE= and SHA256= in the job output); the ordinary fits are backed up to the CFS directory
# named by REQUEST.json (durable_parent).
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:?submit this script from the bundle directory}"
cd "$HERE"
source /global/common/software/desi/desi_environment.sh main
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$HERE/installed"
python -c "import blrfit, numpy, scipy, astropy, sys; print('blrfit', blrfit.__version__, blrfit.__file__); print(sys.version); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'astropy', astropy.__version__)"
python study.py verify
python study.py run --out "$PSCRATCH/SINGLE_UNCERTAINTY_${SLURM_JOB_ID}"
