from transformers.configuration_utils import PretrainedConfig

class SMT5Config(PretrainedConfig):
    model_type = "smt5"

    def __init__(
        self,
        d_out=1024,
        d_proj=768,
        temp=0.05,
        **kwargs,
    ):
        self.d_out = d_out
        self.d_proj = d_proj
        self.temp = temp
