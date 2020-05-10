import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from qurator.sbb_textline_detector.crop import SbbCrop

@click.command()
@ocrd_cli_options
def ocrd_sbb_crop(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbCrop, *args, **kwargs)
