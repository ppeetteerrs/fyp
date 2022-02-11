FROM ppeetteerrs/pytorch_zsh:latest

WORKDIR /workspace

RUN mamba install -y opencv pycuda nibabel pydicom python-lmdb python-dotenv tensorboard pytorch-lightning
RUN mamba install -c rapidsai -c nvidia cusignal
RUN mamba install -y -c simpleitk simpleitk
RUN pip install -U stylegan2_torch deepdrr torch-tb-profiler lpips torchgeometry

CMD "zsh"
