# This file is autogenerated by the command `make fix-copies`, do not edit.
# flake8: noqa

from ..utils import DummyObject, requires_backends


class StableDiffusionKDiffusionPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers", "k_diffusion"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers", "k_diffusion"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "k_diffusion"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "k_diffusion"])
