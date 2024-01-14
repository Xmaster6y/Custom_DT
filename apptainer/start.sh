#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=<yoann.poupart@hhi-extern.fraunhofer.de>
#SBATCH --job-name=apptainer
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used:
# - One for the ImageNet dataset and
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
# apptainer build --fakeroot --force ./apptainer/leela_train.sif ./apptainer/leela_train.def
timeout 12h apptainer run --nv --bind ${LOCAL_JOB_DIR}:/opt/output ./apptainer/leela_train.sif -e 1

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cp -r ${LOCAL_JOB_DIR} ${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}
