# Try not to renew this image unless you know what you are doing
# See: https://github.com/pytorch/pytorch/issues/74437 on why StyleGAN2 might not work in newer PyTorch versions
FROM ghcr.io/ppeetteerrs/ct2cxr_pytorch

# Add User
ARG USERNAME=user
ARG USER_UID
ARG USER_GID=$USER_UID

RUN echo "Adding user... NAME: $USERNAME - UID: $USER_UID - GID: $USER_GID"

RUN groupadd --gid $USER_GID $USERNAME && \
	useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
	echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
	chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME

# Shell
## ZSH
RUN sudo apt-get install -y zsh && \
	sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
	git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
	git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
	echo "export HISTSIZE=1000000" >> ~/.zshrc && \
	echo "export SAVEHIST=1000000" >> ~/.zshrc && \
	echo "setopt EXTENDED_HISTORY" >> ~/.zshrc && \
	sed -i 's/plugins=(git)/plugins=(zsh-syntax-highlighting zsh-autosuggestions)/g' ~/.zshrc

## Starship
RUN /resources/starship.sh

## Aliases
RUN cp /resources/aliases.bashrc ~/.aliases
RUN echo "source ~/.aliases" >> ~/.bashrc
RUN if [ -x "$(command -v zsh)" ]; then echo "source ~/.aliases" >> ~/.zshrc; fi

# Mamba
# Install mamba
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
	bash "Mambaforge-$(uname)-$(uname -m).sh" -b && \
	rm "Mambaforge-$(uname)-$(uname -m).sh" && \
	conda create -p ~/mambaforge/envs/user --clone base -y

## Automatically activate user environment, disable logo and disable conda prompt for starship
ENV PATH=/home/user/mambaforge/bin:$PATH
RUN mamba init --all && \
	conda config --set changeps1 False && \
	echo "export MAMBA_NO_BANNER=1" >> ~/.bashrc && \
	echo "export MAMBA_NO_BANNER=1" >> ~/.zshrc && \
	echo "conda activate user" >> ~/.bashrc && \
	echo "conda activate user" >> ~/.zshrc

## Change shell
SHELL ["conda", "run", "-n", "user", "/bin/bash", "-c"]

# Linters and Formatters
RUN mamba install -n base -y autoflake && \
	mamba install -y black isort flake8 tqdm jupyter notebook rich numpy=1.21.5 scipy matplotlib pandas seaborn

RUN pip install ipympl lpips torchgeometry deepdrr==1.1.0a4 stylegan2-torch simple-parsing ipykernel mkdocs-jupyter mkdocs-material mkdocstrings-python torch-fidelity torch-tb-profiler && pip install git+https://github.com/JoHof/lungmask

RUN mamba install -y python-dotenv python-lmdb pycuda scikit-learn && \
	mamba install -y -c simpleitk simpleitk && \
	mamba install -y -c rapidsai -c nvidia cusignal

# OpenCV with autocomplete
ARG OPENCV_VERSION=4.5.5

RUN cd ~ && \
	git clone https://github.com/opencv/opencv && \
	git -C opencv checkout $OPENCV_VERSION && \
	git clone https://github.com/opencv/opencv_contrib && \
	git -C opencv_contrib checkout $OPENCV_VERSION && \
	mkdir -p ~/opencv/build

RUN cd ~/opencv/build && \
	cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
	-D CPU_ONLY=1 \
    -D WITH_TBB=ON \
	-D BUILD_opencv_dnn=OFF \
	-D BUILD_opencv_cnn_3dobj=OFF \
	-D BUILD_opencv_dnn_modern=OFF \
    -D BUILD_opencv_cudacodec=OFF \
    -D ENABLE_FAST_MATH=1 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
	-D HAVE_opencv_python3=ON \
    -D PYTHON3_NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)") \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D OPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=OFF .. \
	-D BUILD_SHARED_LIBS=OFF && \
	sudo make -j$(nproc) install && \
	sudo ldconfig && \
	cd ~ && \
	sudo rm -rf ~/opencv ~/opencv_contrib && \
	sudo apt-get update -y && \
    sudo apt-get install -y libgl1 libxrender1 libgl1-mesa-glx xvfb

CMD "zsh"