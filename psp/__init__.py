from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

from stylegan2_torch import Generator, Resolution, default_channels
from torch import Tensor, concat, nn

from psp.encoder import Encoder


class pSp(nn.Module):
    def __init__(
        self,
        ckpt: Dict[str, Any],
        *,
        use_mean: bool = True,
        e_in_channel: int = 1,
        e_resolution: Resolution = 256,
        g_resolution: Resolution = 256,
        g_latent_dim: int = 512,
        g_n_mlp: int = 8,
        g_lr_mlp_mult: float = 0.01,
        g_channels: Dict[Resolution, int] = default_channels,
        g_blur_kernel: List[int] = [1, 3, 3, 1],
    ):
        """
        pSp module. Consists of an encoder and a pretrained StyleGAN generator.

        Args:
            ckpt (Dict[str, Any]): PyTorch checkpoint dictionary. Can be pretrained StyleGAN or pSp.
            use_mean (bool, optional): Add latent average to generated style vectors.
            e_in_channel (int, optional): Encoder in_channel.
            e_resolution (Resolution, optional): Encoder resolution.
            g_resolution (Resolution, optional): Generator resolution.
            g_latent_dim (int, optional): Generator latent dimension.
            g_n_mlp (int, optional): Generator mapping network layers.
            g_lr_mlp_mult (float, optional): Generator mapping network lr multiplier.
            g_channels (Dict[Resolution, int], optional): Generator no. of channels at each resolution level.
            g_blur_kernel (List[int], optional): Generator blurring kernel.
        """

        super().__init__()

        # Define architecture
        self.encoder = Encoder(in_channel=e_in_channel, resolution=e_resolution).to(
            "cuda"
        )
        self.decoder = Generator(
            resolution=g_resolution,
            latent_dim=g_latent_dim,
            n_mlp=g_n_mlp,
            lr_mlp_mult=g_lr_mlp_mult,
            channels=g_channels,
            blur_kernel=g_blur_kernel,
        ).to("cuda")

        # Load model checkpoints. Resumed flag indicates whether pSp is pretrained.
        if "g_ema" in ckpt:
            self.decoder.load_state_dict(ckpt["g_ema"], strict=True)
            self.resumed = False
        else:
            self.encoder.load_state_dict(ckpt["encoder"], strict=True)
            self.decoder.load_state_dict(ckpt["decoder"], strict=True)
            self.resumed = True

        # Load latent average
        self.latent_avg: Optional[Tensor]
        if use_mean:
            if "latent_avg" in ckpt:
                self.latent_avg = ckpt["latent_avg"].to("cuda")
            else:
                self.latent_avg = self.decoder.mean_latent(10000, device="cuda")
        else:
            self.latent_avg = None

    @overload
    def forward(
        self,
        input: Union[Tensor, Tuple[Tensor, Tensor]],
        task: Literal["encode"],
        mix_mode: Literal["alt", "half", "mean"] = "mean",
        disable_noise: bool = False,
    ) -> Tensor:
        pass

    @overload
    def forward(
        self,
        input: Union[Tensor, Tuple[Tensor, Tensor]],
        task: Literal["generate"],
        mix_mode: Literal["alt", "half", "mean"] = "mean",
        disable_noise: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        pass

    def forward(
        self,
        input: Union[Tensor, Tuple[Tensor, Tensor]],
        task: Literal["encode", "generate"] = "generate",
        mix_mode: Literal["alt", "half", "mean"] = "mean",
        disable_noise: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Performs forward propagation. Task can be encode or generate.

        `encode` returns the style vectors `Tensor`.

        `generate` returns a `Tuple[Tensor, Tensor]` of generated image and style vectors.

        `mix_mode` determines the mixing method when input contains more than 1 image.

        Args:
            input (Union[Tensor, Tuple[Tensor, Tensor]]): Input image(s).
            task (Literal["encode", "generate"], optional): pSp forward task.
            mix_mode (Literal["alt", "half", "mean"], optional): Mixing mode if input contains 2 images.
            disable_noise (bool, optional): Disable noise in StyleGAN.

        Raises:
            NotImplementedError: _description_

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: _description_
        """

        # Mix if two inputs
        if isinstance(input, Tensor):
            codes = self.encoder(input)
        else:
            codes1 = self.encoder(input[0])
            codes2 = self.encoder(input[1])
            codes = self.mix_codes(codes1, codes2, mix_mode=mix_mode)

        # Add latent average if needed
        if self.latent_avg is not None:
            codes += self.latent_avg.repeat(codes.shape[0], 1, 1)

        # Skip image generation if task is encode
        if task == "encode":
            return codes
        elif task == "generate":
            if disable_noise:
                noises = [0] * self.decoder.n_w_plus
            else:
                noises = None
            imgs = self.decoder(
                [codes], return_latents=False, input_type="w_plus", noises=noises
            )
        else:
            raise NotImplementedError(f"pSp task {task} not implemented.")

        return imgs, codes

    @staticmethod
    def mix_codes(
        codes1: Tensor,
        codes2: Tensor,
        mix_mode: Literal["alt", "half", "mean"] = "mean",
    ) -> Tensor:
        """
        Mixes two latent vectors.

        `alt`: Alternates between each latent vector.
        `half`: Takes first half from codes1 and next half from codes2.
        `mean`: Takes the mean of the latent vectors.
        """
        if mix_mode == "alt":
            return concat(
                [
                    codes1[:, i : i + 1, :] if i % 2 == 0 else codes2[:, i : i + 1, :]
                    for i in range(codes1.shape[1])
                ],
                dim=1,
            )
        elif mix_mode == "half":
            idx_half = codes1.shape[1] // 2
            return concat(
                [codes1[:, :idx_half, :], codes2[:, idx_half:, :]],
                dim=1,
            )
        elif mix_mode == "mean":
            return (codes1 + codes2) / 2
        else:
            raise NotImplementedError(f"Mixing mode {mix_mode} not implemented.")
