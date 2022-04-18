FROM ghcr.io/ppeetteerrs/fyp:server

RUN pip install -U stylegan2-torch simple-parsing ipykernel mkdocs-jupyter mkdocs-material mkdocstrings-python && \
	mamba install -y wandb

COPY .wandb_key /home/user

RUN echo "export WANDB_API_KEY=$(cat /home/user/.wandb_key)" >> ~/.zshrc && \
	echo "export WANDB_API_KEY=$(cat /home/user/.wandb_key)" >> ~/.bashrc

CMD "zsh"