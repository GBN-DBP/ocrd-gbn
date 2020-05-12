import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from qurator.sbb_textline_detector.crop import SbbCrop
from qurator.sbb_textline_detector.segment_page import SbbSegmentPage

@click.command()
@ocrd_cli_options
def ocrd_sbb_crop(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbCrop, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_sbb_segment_page(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbSegmentPage, *args, **kwargs)
