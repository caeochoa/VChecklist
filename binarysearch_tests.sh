#!/usr/bin/env bash

# test the first half perturbed

# sbatch --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Half1Perturbed tests_full_data.sh 0 75 PredictPerturbTest # Doesn't work

# test the second half perturbed 

# sbatch --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Half2Perturbed tests_full_data.sh 76 150 PredictPerturbTest # Doesn't work

# test four consecutive perturbed

# sbatch --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Conseq4Perturbed tests_full_data.sh 30 33 PredictPerturbTest # works

# test four non-consecutive pertubed

# sbatch --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J NoConseq4Perturbed --begin="14:42 07/09/22" tests_full_data.sh 0 5 PredictPerturbTest # works

# test quarters

# sbatch --begin="15:15 07/09/22" --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q1Perturbed tests_full_data.sh 0 37 PredictPerturbTest # works very slowly?

# sbatch --begin="15:30 07/09/22" --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q2Perturbed tests_full_data.sh 38 75 PredictPerturbTest # works

# sbatch --begin="15:45 07/09/22" --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q3Perturbed tests_full_data.sh 75 112 PredictPerturbTest # works

# sbatch --begin="16:00 07/09/22" --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q4Perturbed tests_full_data.sh 112 150 PredictPerturbTest # works

# test last three quarters

# sbatch --mail-type=ALL --mail-user=s2259310 --begin="15:15 07/09/22" --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q2-4Perturbed tests_full_data.sh 38 150 PredictPerturbTest

# test first two quarters with added verbosity to nnunet

# sbatch --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q1Perturbed tests_full_data.sh 0 37 PredictPerturbTest # 

# sbatch --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q2Perturbed tests_full_data.sh 38 75 PredictPerturbTest # 10 mins aprox

# test second two quarters and middle two quarters together

# sbatch --exclude="landonia02 landonia04 landonia08" --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q3Perturbed tests_full_data.sh 75 112 PredictPerturbTest # 10 mins aprox

# sbatch --exclude="landonia02 landonia04 landonia05 landonia08" --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q4Perturbed tests_full_data.sh 112 150 PredictPerturbTest #

# sbatch --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J Q2-3Perturbed tests_full_data.sh 38 112 PredictPerturbTest # 20 mins aprox

# divide q1 and q4 in half and test

# sbatch --exclude="" --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J HQ11Perturbed tests_full_data.sh 0 18 PredictPerturbTest

# sbatch --exclude="landonia02" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J HQ12Perturbed tests_full_data.sh 19 37 PredictPerturbTest

# sbatch --exclude="landonia02 landonia04 landonia05 landonia07 landonia08" --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J HQ41Perturbed tests_full_data.sh 112 130 PredictPerturbTest

# sbatch --exclude="landonia02 landonia04 landonia05 landonia06 landonia07 landonia08" --mail-type=ALL --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J HQ42Perturbed tests_full_data.sh 131 150 PredictPerturbTest

# divide q42 in half and test

# sbatch --exclude="landonia02 landonia07" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 421Perturbed tests_full_data.sh 131 140 PredictPerturbTest
# sbatch --exclude="landonia02 landonia07 landonia08" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 422Perturbed tests_full_data.sh 141 150 PredictPerturbTest

# divide 421 and 11

# sbatch --exclude="" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 4211Perturbed tests_full_data.sh 131 135 2
# sbatch --exclude="landonia02" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 4212Perturbed tests_full_data.sh 136 140 2

# sbatch --exclude="landonia02 landonia07" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 111Perturbed tests_full_data.sh 0 9 2
# sbatch --exclude="landonia02 landonia07 landonia08" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 112Perturbed tests_full_data.sh 10 18 2

# divide 4211 and 111

# sbatch --nodelist="landonia07" --exclude="" --mail-type=END --mail-user=s2259310 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 42111Perturbed tests_full_data.sh 131 133 2
# sbatch --nodelist="landonia04" --exclude="" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 42112Perturbed tests_full_data.sh 134 135 2

# sbatch --nodelist="landonia05" --exclude="" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 1111Perturbed tests_full_data.sh 0 5 2
# sbatch --nodelist="landonia06" --exclude="" --mail-type=END --mail-user=s2259310 --time=08:00:00 --mem=14000 --cpus-per-task=4 --gres=gpu:1 -J 1112Perturbed tests_full_data.sh 6 9 2

# let's try again 111 but a min gpu mem limit

# sbatch  --exclude="" --mail-type=END --mail-user=s2259310 --cpus-per-task=4 --gres=gpu:gtx-1060:1 -J 111Perturbed tests_full_data.sh 0 9 2

# let's try Q4 with a titan-x

# sbatch --mail-type=END --mail-user=s2259310  --mem=14000 --cpus-per-task=4 --gres=gpu:titan-x:1 -J Q4Perturbed tests_full_data.sh 112 150 PredictPerturbTest

# and on a a6000?

sbatch --mail-type=END --mail-user=s2259310  --mem=14000 --cpus-per-task=4 --gres=gpu:a6000:1 -J Q4Perturbed tests_full_data.sh 112 150 PredictPerturbTest
