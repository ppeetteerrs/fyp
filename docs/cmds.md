## Experiments

### Loss Function Ablation

#### Ground Truth: CT projection (input)

| name       | l2 loss    | reg loss | lpips loss | id loss | observations                                                     |
| ---------- | ---------- | -------- | ---------- | ------- | ---------------------------------------------------------------- |
| psp_l2     | out:body:3 | 0        | -          | -       | failed: low fidelity, rough structural preservation              |
| psp_l2_reg | out:body:3 | 0.01     | -          | -       | failed: high fidelity, low pathology and structural preservation |

#### Ground Truth: Soft tissues projection (bone extraction method in paper)

| name                                 | l2 loss                   | reg loss | lpips loss   | id loss      | observations                                                             |
| ------------------------------------ | ------------------------- | -------- | ------------ | ------------ | ------------------------------------------------------------------------ |
| psp_l2_soft_reg                      | out:soft:3                | 0.01     | -            | -            | failed: serious artifacts                                                |
| psp_l2_soft_reg_lpips (2 tries)      | out:soft:3                | 0.01     | out:body:0.1 | -            | failed: blurry outputs                                                   |
| psp_l2_soft_less_reg_lpips (2 tries) | out:soft:1                | 0.01     | out:body:0.1 | -            | success: outputs look fine, might need to tune reg  loss to reduce bones |
| psp_l2_soft_reg_lpips_id             | out:soft:3                | 0.01     | out:body:0.1 | out:body:0.8 | failed: blurry outputs                                                   |
| psp_l2_soft_mix_reg_lpips            | out:soft:2.5,out:body:0.5 | 0.01     | out:body:0.1 | -            | failed: blurry outputs                                                   |
| psp_l2_soft_mix_less_reg_lpips       | out:soft:1,out:body:0.5   | 0.01     | out:body:0.1 | -            | success: outputs look fine                                               |


(reason for soft_reg_lpips to fail: suspect that normalization method was changed so paper results are not reproducible)
(I should use mix without LPIPS)

#### Ground Truth: Lung tissues projection (lung segmentation)

| name                           | l2 loss                   | reg loss | lpips loss   | id loss      | observations                                           |
| ------------------------------ | ------------------------- | -------- | ------------ | ------------ | ------------------------------------------------------ |
| psp_l2_lung_reg                | out:lung:3                | 0.01     | -            | -            | failed: outputs become lung tissues                    |
| psp_l2_lung_reg_lpips          | out:lung:3                | 0.01     | out:body:0.1 | -            | success: outputs look good, bones not overly prominent |
| psp_l2_lung_reg_lpips_id       | out:lung:3                | 0.01     | out:body:0.1 | out:body:0.8 | failed: outputs become lung tissues                    |
| psp_l2_lung_mix_reg_lpips      | out:lung:2.5,out:body:0.5 | 0.01     | out:body:0.1 | -            | failed: a blurry mess                                  |
| psp_l2_lung_mix_less_reg_lpips | out:lung:1,out:body:0.5   | 0.01     | out:body:0.1 | -            | failed: a blurry mess                                  |

### psp_l2

`python run.py --name psp_l2 psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:body:3 --id "" --lpips "" --reg 0`

`python run.py --name psp_l2 psp --ckpt output/psp_l2/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_reg

`python run.py --name psp_l2_reg psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:body:3 --id "" --lpips "" --reg 0.01`

`python run.py --name psp_l2_reg psp --ckpt output/psp_l2_reg/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_soft_reg

`python run.py --name psp_l2_soft_reg psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:3 --id "" --lpips "" --reg 0.01`

`python run.py --name psp_l2_soft_reg psp --ckpt output/psp_l2_soft_reg/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_soft_reg_lpips

`python run.py --name psp_l2_soft_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:3 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_soft_reg_lpips psp --ckpt output/psp_l2_soft_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_soft_reg_lpips2

`python run.py --name psp_l2_soft_reg_lpips2 psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:3 --id "" --lpips "out:body:0.1" --reg 0.01`

### psp_l2_soft_less_reg_lpips

`python run.py --name psp_l2_soft_less_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:1 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_soft_less_reg_lpips psp --ckpt output/psp_l2_soft_less_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

`python run.py --name psp_l2_soft_less_reg_lpips psp --ckpt output/psp_l2_soft_less_reg_lpips/200000.pt mix --dataset input/data/covid_ct_lmdb --n_images 2000 --mix_mode mean`

### psp_l2_soft_less_reg_lpips2

`python run.py --name psp_l2_soft_less_reg_lpips2 psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:1 --id "" --lpips "out:body:0.1" --reg 0.01`

### psp_l2_soft_reg_lpips_id

`python run.py --name psp_l2_soft_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:3 --id "out:body:0.8" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_soft_reg_lpips_id psp --ckpt output/psp_l2_soft_reg_lpips_id/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_soft_mix_reg_lpips

`python run.py --name psp_l2_soft_mix_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:2.5,out:body:0.5 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_soft_mix_reg_lpips psp --ckpt output/psp_l2_soft_mix_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_soft_mix_less_reg_lpips

`python run.py --name psp_l2_soft_mix_less_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:soft:1,out:body:0.5 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_soft_mix_less_reg_lpips psp --ckpt output/psp_l2_soft_mix_less_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_lung_reg

`python run.py --name psp_l2_lung_reg psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:lung:3 --id "" --lpips "" --reg 0.01`

`python run.py --name psp_l2_lung_reg psp --ckpt output/psp_l2_lung_reg/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_lung_reg_lpips

`python run.py --name psp_l2_lung_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:lung:3 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_lung_reg_lpips psp --ckpt output/psp_l2_lung_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

`python run.py --name psp_l2_lung_reg_lpips psp --ckpt output/psp_l2_lung_reg_lpips/200000.pt mix --dataset input/data/covid_ct_lmdb --n_images 2000 --mix_mode mean`

### psp_l2_lung_reg_lpips2

`python run.py --name psp_l2_lung_reg_lpips2 psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:lung:3 --id "" --lpips "out:body:0.1" --reg 0.01`

### psp_l2_lung_reg_lpips_id

`python run.py --name psp_l2_lung_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:lung:3 --id "out:body:0.8" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_lung_reg_lpips_id psp --ckpt output/psp_l2_lung_reg_lpips_id/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_lung_mix_reg_lpips

`python run.py --name psp_l2_lung_mix_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:lung:2.5,out:body:0.5 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_lung_mix_reg_lpips psp --ckpt output/psp_l2_lung_mix_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

### psp_l2_lung_mix_less_reg_lpips

`python run.py --name psp_l2_lung_mix_less_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset input/data/covid_ct_lmdb --l2 out:lung:1,out:body:0.5 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_lung_mix_less_reg_lpips psp --ckpt output/psp_l2_lung_mix_less_reg_lpips/200000.pt generate --dataset input/data/covid_ct_lmdb`

tests:
- l2 (out vs body) -- done
- l2 (out vs body) + reg -- todo
- l2 (out vs soft) + reg -- done
- l2 (out vs lung) + reg -- running
- l2 (out vs soft) + reg + lpips -- running
- l2 (out vs lung) + reg + lpips -- running
- l2 (out vs soft) + reg + lpips + id -- running
- l2 (out vs lung) + reg + lpips + id -- todo

Then test:
- can preserve small pneumonia? soft and lung


Same analysis for lung segmentation according to my paper

 python run.py --name psp_l2_reg psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:body:3 --id "" --lpips ""  --reg 0.01

 python run.py --name psp_l2_lung_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:lung:3 --id "" --lpips out:body:0.1  --reg 0.01


 python run.py --name psp_l2_soft_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:soft:3 --id "" --lpips out:body:0.1  --reg 0.01

  python run.py --name psp_l2_soft_less_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:soft:1 --id "" --lpips out:body:0.1  --reg 0.01


python run.py --name psp_l2_soft_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:soft:3 --id out:body:0.8 --lpips out:body:0.1  --reg 0.01

TODO:::
python run.py --name psp_l2_lung_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:lung:3 --id out:body:0.8 --lpips out:body:0.1  --reg 0.01


python run.py --name psp_l2_soft_mix_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:soft:2.5,out:body:0.5 --id "" --lpips out:body:0.1  --reg 0.01


python run.py --name psp_l2_lung_mix_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --datasets input/data/covid_ct_lmdb --l2 out:lung:2.5,out:body:0.5 --id "" --lpips out:body:0.1  --reg 0.01