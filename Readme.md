## requirments
extended NNI package from the repo:
https://github.com/XuYongi/nni/tree/v1.7.1
install NNi1.7.1 by sourcecode

torch==1.5

## 1. Main Process
### train original model

python train_dice_bigbitchsize.py
or
python train_dice_bigbitchsize_bceloss.py

### prune and train model
python train_dice_bigbitchsize_pruner.py

### speedup and save pruned model
python speedup.py

### load pruned model to pridict
python pridict_speedup.py


## 2. More Tests
### compare the inference time between use mask and speedup 
python speedup_test.py

### a pruner pipeline
model_pruner.py
