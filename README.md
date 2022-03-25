
### Generate Docs
```bash
cp README.md docs/index.md
mkdocs gh-deploy --force
```


```bash
cp README.md docs/index.md
mkdocs serve
```
Train styleGAN

```bash
torchrun --nproc_per_node=4 run.py stylegan train
```