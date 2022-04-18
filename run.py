from scripts.psp.generate import psp_generate
from scripts.psp.mix import psp_mix
from scripts.psp.train import psp_train
from scripts.stylegan.generate import stylegan_generate
from scripts.stylegan.train import stylegan_train
from utils.cli import OPTIONS
from utils.cli.psp import PSPArch, PSPGenerate, PSPMix, PSPTrain
from utils.cli.stylegan import StyleGANArch, StyleGANGenerate, StyleGANTrain

if __name__ == "__main__":
    if isinstance(OPTIONS.arch, StyleGANArch):
        if isinstance(OPTIONS.arch.cmd, StyleGANTrain):
            stylegan_train()
        elif isinstance(OPTIONS.arch.cmd, StyleGANGenerate):
            stylegan_generate()
        else:
            raise Exception("Invalid CLI arguments.")
    elif isinstance(OPTIONS.arch, PSPArch):
        if isinstance(OPTIONS.arch.cmd, PSPTrain):
            psp_train()
        elif isinstance(OPTIONS.arch.cmd, PSPGenerate):
            psp_generate()
        elif isinstance(OPTIONS.arch.cmd, PSPMix):
            psp_mix()
        else:
            raise Exception("Invalid CLI arguments.")
    else:
        raise Exception("Invalid CLI arguments.")
