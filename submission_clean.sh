#! /bin.bash
#SBATCH --array=0-215
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00
#SBATCH -N 1
#SBATCH -p general
#SBATCH --mem-per-cpu=20g

module load python
python3 sentiAnalysis_clean.py --wikiModelDir . --dataFileList dataFileArray/dataFileArray$SLURM_ARRAY_TASK_ID.txt --dataFileDir PARSED --cpus 1
