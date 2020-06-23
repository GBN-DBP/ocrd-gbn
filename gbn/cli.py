import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from gbn.sbb.predict import OcrdGbnSbbPredict
from gbn.sbb.binarize import OcrdGbnSbbBinarize
from gbn.sbb.crop import OcrdGbnSbbCrop
from gbn.sbb.segment import OcrdGbnSbbSegment

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_predict(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdGbnSbbPredict, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_binarize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdGbnSbbBinarize, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_crop(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdGbnSbbCrop, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_gbn_sbb_segment(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdGbnSbbSegment, *args, **kwargs)
