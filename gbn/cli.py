import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from gbn.sbb.predict import Predict
from gbn.sbb.page_segment import PageSegment
from gbn.sbb.region_segment import RegionSegment

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_predict(*args, **kwargs):
    return ocrd_cli_wrap_processor(Predict, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_page_segment(*args, **kwargs):
    return ocrd_cli_wrap_processor(PageSegment, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_region_segment(*args, **kwargs):
    return ocrd_cli_wrap_processor(RegionSegment, *args, **kwargs)
