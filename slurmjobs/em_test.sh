#!/bin/bash
#SBATCH --job-name=tomato
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=30G
#SBATCH --time=36:00:00
#SBATCH --output=/cwork/jfr29/Sepy/output/em_encounter_%A_%a.out
#SBATCH --error=/cwork/jfr29/Sepy/errors/em_encounter_%A_%a.err

YEARS=(2014)
CONFIGURATION_FILE=/cwork/jfr29/Sepy/configurations/emory_config.yaml

#### FOR LOOP: takes each year and divides the patients in chunks based on number of cores ###
for year in ${YEARS[*]}
do
echo Currently processing year: $year

python /cwork/jfr29/Sepy/make_dicts.py $year $CONFIGURATION_FILE

done