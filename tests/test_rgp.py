"""Tests for the Raccoon Game Protocol."""

import io

from raccoon.model.network import RaccoonNet
from raccoon.protocol.rgp import RGPEngine


def test_rgp_identify():
    engine = RGPEngine(RaccoonNet())
    inp = io.StringIO("rgp\nquit\n")
    out = io.StringIO()
    engine.run(input_stream=inp, output_stream=out)
    output = out.getvalue()
    assert "id name Raccoon" in output
    assert "rgpok" in output


def test_rgp_isready():
    engine = RGPEngine(RaccoonNet())
    inp = io.StringIO("isready\nquit\n")
    out = io.StringIO()
    engine.run(input_stream=inp, output_stream=out)
    assert "readyok" in out.getvalue()


def test_rgp_newgame_and_go():
    engine = RGPEngine(RaccoonNet(), default_simulations=5)
    inp = io.StringIO("newgame\ngo simulations 5\nquit\n")
    out = io.StringIO()
    engine.run(input_stream=inp, output_stream=out)
    output = out.getvalue()
    assert "bestmove" in output


def test_rgp_quit():
    engine = RGPEngine(RaccoonNet())
    inp = io.StringIO("quit\n")
    out = io.StringIO()
    engine.run(input_stream=inp, output_stream=out)
    assert not engine.running
