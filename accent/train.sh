#!/bin/bash
for fold in {0..4}
do
    python train_fold.py --fold ${fold} --exp 8 --seed 34
done

python train_fold.py --make_sub 'sub8.csv'