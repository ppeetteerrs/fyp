## Images
- `body`: naiive CT projection of body (i.e. bed removed)
- `soft`: soft tissues projection (i.e. bones removed)
- `lung`: lung projection (lung obtained via [lung segmentation](https://github.com/JoHof/lungmask))
- `out`: model output CXR image

## Loss Function
- `L2`: pixel-wise $L_2$ loss between target and ground truth images, syntax: `target:truth:weight`
- `Reg`: regularization loss, norm of encoder output extended latent vector (does not include mean latent vector), syntax: `weight`
- `LPIPS`: LPIPS perceptual similarity loss, syntax: `target:truth:weight`
- `ID`: Arcface facial identity loss in original pSp paper, syntax: `target:truth:weight`

## Ablation Study

We first start with the unmodified pSp loss function and investigate the effect of each loss term.

#### Ground Truth: CT projection (input)

| name                | $L_2$      | Reg  | LPIPS        | ID           | outcome                                                   | fidelity | pathology and structural preservation |
| ------------------- | ---------- | ---- | ------------ | ------------ | --------------------------------------------------------- | -------- | ------------------------------------- |
| `body`              | out:body:1 | 0    | -            | -            | <span style="color: red; font-weight: 700;">FAILED</span> |          |                                       |
| `body_reg`          | out:body:1 | 0.01 | -            | -            | <span style="color: red; font-weight: 700;">FAILED</span> |          |                                       |
| `body_reg_lpips`    | out:body:1 | 0.01 | out:body:0.1 | -            | <span style="color: red; font-weight: 700;">FAILED</span> |          |                                       |
| `body_reg_lpips_id` | out:body:1 | 0.01 | out:body:0.1 | out:body:0.3 | <span style="color: red; font-weight: 700;">FAILED</span> |          |                                       |

#### Ground Truth: Soft tissues projection (bone extraction method in paper)


| name                  | $L_2$                   | Reg  | LPIPS        | ID           | outcome | fidelity | pathology and structural preservation |
| --------------------- | ----------------------- | ---- | ------------ | ------------ | ------- | -------- | ------------------------------------- |
| `soft_reg`            | out:soft:1              | 0.01 | -            | -            |         |          |                                       |
| `soft_reg_lpips`      | out:soft:1              | 0.01 | out:body:0.1 | -            |         |          |                                       |
| `soft_reg_lpips_id`   | out:soft:1              | 0.01 | out:body:0.1 | out:body:0.3 |         |          |                                       |
| `soft_body_reg`       | out:soft:1,out:body:0.5 | 0.01 | -            | -            |         |          |                                       |
| `soft_body_reg_lpips` | out:soft:1,out:body:0.5 | 0.01 | out:body:0.1 | -            |         |          |                                       |

#### Ground Truth: Lung tissues projection (lung segmentation)


| name                  | $L_2$                   | Reg  | LPIPS        | ID           | outcome | fidelity | pathology and structural preservation |
| --------------------- | ----------------------- | ---- | ------------ | ------------ | ------- | -------- | ------------------------------------- |
| `lung_reg`            | out:lung:1              | 0.01 | -            | -            |         |          |                                       |
| `lung_reg_lpips`      | out:lung:1              | 0.01 | out:body:0.1 | -            |         |          |                                       |
| `lung_reg_lpips_id`   | out:lung:1              | 0.01 | out:body:0.1 | out:body:0.3 |         |          |                                       |
| `lung_body_reg`       | out:lung:1,out:body:0.5 | 0.01 | -            | -            |         |          |                                       |
| `lung_body_reg_lpips` | out:lung:1,out:body:0.5 | 0.01 | out:body:0.1 | -            |         |          |                                       |


### psp_l2_soft_less_reg_lpips

`python run.py --name soft_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:soft:1 --id "" --lpips "out:body:0.1" --reg 0.01`

`python run.py --name psp_l2_soft_less_reg_lpips psp --ckpt output/psp_l2_soft_less_reg_lpips/200000.pt generate --dataset output/covid_ct/lmdb`

`python run.py --name psp_l2_soft_less_reg_lpips psp --ckpt output/psp_l2_soft_less_reg_lpips/200000.pt mix --dataset output/covid_ct/lmdb --n_images 2000 --mix_mode mean`


RUN 1

python run.py --name soft_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:soft:1 --id "" --lpips out:body:0.1 --reg 0.01
python run.py --name soft_body_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:soft:1,out:body:0.5 --id "" --lpips out:body:0.1 --reg 0.01
python run.py --name lung_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:lung:1 --id "" --lpips out:body:0.1 --reg 0.01
python run.py --name lung_body_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:lung:1,out:body:0.5 --id "" --lpips out:body:0.1 --reg 0.01

RUN 2

python run.py --name soft_reg psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:soft:1 --id "" --lpips "" --reg 0.01
python run.py --name soft_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:soft:1 --id out:body:0.3 --lpips out:body:0.1 --reg 0.01
python run.py --name lung_reg psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:lung:1 --id "" --lpips "" --reg 0.01
python run.py --name lung_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:lung:1 --id out:body:0.3 --lpips out:body:0.1 --reg 0.01

RUN 3

python run.py --name soft_body_reg psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:soft:1,out:body:0.5 --id "" --lpips "" --reg 0.01
python run.py --name lung_body_reg psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:lung:1,out:body:0.5 --id "" --lpips "" --reg 0.01
python run.py --name body psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:body:1 --id "" --lpips "" --reg 0
python run.py --name body_reg psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:body:1 --id "" --lpips "" --reg 0.01

RUN 4

python run.py --name body_reg_lpips psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:body:1 --id "" --lpips out:body:0.1 --reg 0.01
python run.py --name body_reg_lpips_id psp --ckpt input/pretrained/stylegan.pt train --dataset output/covid_ct/lmdb --l2 out:body:1 --id out:body:0.3 --lpips out:body:0.1 --reg 0.01