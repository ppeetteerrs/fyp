import os
from pathlib import Path

project_root = Path(__file__).parents[2].resolve()
if Path(os.getcwd()) != project_root:
    print(f"Please run the script from {project_root}. Exiting...")
    exit(1)


from dataclasses import dataclass
from os import environ
from typing import Optional, Union

from IPython import get_ipython
from simple_parsing import ArgumentParser, Serializable, field, subparsers
from utils.cli.psp import PSPArch
from utils.cli.stylegan import StyleGANArch


@dataclass
class Options(Serializable):
    """
    Project Options
    """

    arch: Union[StyleGANArch, PSPArch] = subparsers(
        {"stylegan": StyleGANArch, "psp": PSPArch}
    )

    name: str = "default"  # Experiment name
    file: Optional[Path] = field(alias="f", default=None)

    @property
    def output_dir(self) -> Path:
        return Path("output") / self.name


parser = ArgumentParser()
parser.add_arguments(Options, dest="options")

if "options" in environ:
    options_path = Path(environ["options"])
    print(f"Loading options from {options_path.resolve()}...")
    OPTIONS = Options.load(options_path, drop_extra_fields=True)

else:
    if get_ipython() is None:
        OPTIONS: Options = parser.parse_args().options

        if OPTIONS.file is not None:
            print(f"Loading options from {OPTIONS.file.resolve()}...")
            OPTIONS = Options.load(OPTIONS.file, drop_extra_fields=True)
    else:
        OPTIONS = Options()

OPTIONS.output_dir.mkdir(parents=True, exist_ok=True)
OPTIONS.dump_yaml(open(OPTIONS.output_dir / "options.yaml", "w"))
