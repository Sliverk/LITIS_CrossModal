# Cross-Modal Verification (CMV) model
Accepted by ESANN in 2021 
[Paper](https://www.esann.org/sites/default/files/proceedings/2021/ES2021-97.pdf)
[AVOR](https://github.com/Sliverk/AVOR) Dataset

## Step 1. Train CMV model with AVOR
CHANE THE FILE PATH FIRST !!!
```bash
python3 0203_train_lenet5_2cls.py
```

## Step 2. Get the CMV model runing on validation split
CHANE THE FILE PATH FIRST !!!
```bash
python3 0304_classification_cnn5_2cls.py
```

## Step 3. Calculate Average Precision
CHANE THE FILE PATH FIRST !!!
```bash
python3 0401_get_mAP.py
```

## Step 4. (Optional) Calculate Number of False Positives
CHANE THE FILE PATH FIRST !!!
```bash
0501_get_cls_fp.py
```
