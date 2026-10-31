"""Original experimental policy on top of the attributed external contributor runtime.

Research-only: the base's kernel implementations are third-party code, properly
attributed, and may NOT be submitted/claimed as our original Triton kernels.
Independent contribution: per-layer exact MLP replay policy + NF4 boundary budget.
"""
_ELIGIBLE = frozenset(range(20))
_INSTALLS = 0

def install(replay_layers=20, split_forward=2, split_backward=3):
    global _INSTALLS
    from transformers.models.qwen2 import modeling_qwen2
    from . import mlp_save_up, qgroup
    if getattr(modeling_qwen2.Qwen2DecoderLayer.__init__, "_hybrid_tagged_layers", False):
        return True
    if not (1 <= replay_layers <= 28 and 1 <= split_forward <= 4 and 1 <= split_backward <= 5):
        raise ValueError("Unsafe research budget")
    cls = modeling_qwen2.Qwen2DecoderLayer
    previous_init = cls.__init__
    previous_eligible = mlp_save_up.save_up_group

    def tagged_init(self, *args, **kwargs):
        previous_init(self,*args,**kwargs)
        idx = getattr(getattr(self,"self_attn",None),"layer_idx",None)
        if type(idx) is int and 0 <= idx < 28:
            self.mlp._research_mlp_layer = idx

    def eligible_group(mlp,x):
        idx = getattr(mlp,"_research_mlp_layer",None)
        if type(idx) is not int or not (0 <= idx < replay_layers):
            return None
        return previous_eligible(mlp,x)

    tagged_init._hybrid_tagged_layers = True
    cls.__init__ = tagged_init
    mlp_save_up.save_up_group = eligible_group
    qgroup.SPLIT_LAYERS["fwd"] = split_forward
    qgroup.SPLIT_LAYERS["bwd"] = split_backward
    _INSTALLS += 1
    return True
