from typing import Any, Dict, Optional, Tuple

import torch
from stylegan2_torch import Discriminator, Generator
from torch import Tensor, nn
from utils.config import CONFIG

from psp.encoder import Encoder, EncoderDeep
from psp.merger import Merger


class pSp(nn.Module):
    def __init__(self, ckpt: Dict[str, Any]):
        super().__init__()

        # Define architecture
        self.encoder = Encoder(CONFIG.PSP_IN_CHANNEL, CONFIG.RESOLUTION).to("cuda")

        self.decoder = Generator(
            CONFIG.RESOLUTION,
            CONFIG.LATENT_DIM,
            CONFIG.N_MLP,
            CONFIG.LR_MLP_MULT,
            CONFIG.STYLEGAN_CHANNELS,
            CONFIG.BLUR_KERNEL,
        ).to("cuda")

        if CONFIG.PSP_LOSS_ID_DISCRIMINATOR > 0 or CONFIG.PSP_LOSS_DISCRIMINATOR > 0:
            self.discriminator = Discriminator(
                CONFIG.RESOLUTION, CONFIG.STYLEGAN_CHANNELS, CONFIG.BLUR_KERNEL
            ).to("cuda")
        else:
            self.discriminator = None

        # Load model checkpoints
        if "g_ema" in ckpt:
            self.decoder.load_state_dict(ckpt["g_ema"], strict=True)
            if self.discriminator is not None:
                self.discriminator.load_state_dict(ckpt["d"], strict=True)
            self.resumed = False
        else:
            self.encoder.load_state_dict(ckpt["encoder"], strict=True)
            self.decoder.load_state_dict(ckpt["decoder"], strict=True)
            if self.discriminator is not None:
                self.discriminator.load_state_dict(ckpt["discriminator"], strict=True)
            self.resumed = True

        # Load latent average
        self.latent_avg: Optional[Tensor]
        if CONFIG.PSP_USE_MEAN:
            if "latent_avg" in ckpt:
                self.latent_avg = ckpt["latent_avg"]
            else:
                self.latent_avg = self.decoder.mean_latent(10000, device="cuda")
        else:
            self.latent_avg = None

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        # Input is n_channel image

        codes = self.encoder(input)

        if self.latent_avg is not None:
            codes += self.latent_avg.repeat(codes.shape[0], 1, 1)

        # No latent mask or injection
        styled_images = self.decoder(
            [codes], return_latents=False, input_type="w_plus", noises=None
        )

        return styled_images, codes
