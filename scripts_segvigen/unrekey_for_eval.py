"""Reverse of rekey_full_seg.py: turn a lightning-framework denoiser
checkpoint back into the standalone SegviGen-emissive format so the existing
evaluation pipeline can load it unchanged.

Why this exists: lightning writes Lightning-format checkpoints (a big dict
with epoch/optimizer/scheduler/ema/... state, model weights nested under
"state_dict"); the evaluation pipeline
(segvigen_emissive/emissive/eval/eval_emissive.py:291-293) loads the
STANDALONE format instead:

    sd = torch.load(args.ckpt, map_location=device)["state_dict"]
    sd = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in sd.items()])
    gen.load_state_dict(sd)   # gen = Gen3DSeg(flow_model), i.e. gen.flow_model.<key>

So the on-disk format eval expects is a top-level {"state_dict": {...}} whose
every key is prefixed "gen3dseg.flow_model." -- verified 2026-08-26 against a
real production checkpoint
(segvigen_emissive/outputs/emis_agentic37k_from72kv2_pw1b/epoch_0008.ckpt):
640 keys, every one prefixed "gen3dseg.flow_model.". Without this bridge, no
lightning-trained checkpoint can enter that eval lane.

Accepts three input shapes for the denoiser's bare (unprefixed) state dict:

  1. A lightning training checkpoint: top-level "state_dict" holds
     "models.denoiser.<key>" tensors (self.models = nn.ModuleDict({"denoiser":
     ...}) in trellis2/trainers/basic.py). This is the default source.
     NOTE (verified 2026-08-26 on a real smoke checkpoint): the same
     state_dict ALSO carries "training_models.denoiser.<key>" duplicates --
     byte-identical to "models.denoiser.<key>" for all 640 tensors in the
     checkpoint checked. These are register-time aliases, not a second set
     of weights; this script uses "models.denoiser." and ignores
     "training_models.denoiser." (loudly, see below).
  2. --ema RATE: instead of the raw weights, pull the EMA shadow from
     ckpt["ema"]["state"][str(RATE)] (same "models.denoiser.<key>" keying,
     640 keys, confirmed 2026-08-26). Mirrors the standalone lane's
     epoch_XXXX.ckpt vs epoch_XXXX_ema.ckpt split.
  3. An already-bare rekey (rekey_full_seg.py's own output format):
     {"denoiser": {<key>: tensor}}. No --ema support for this input shape
     (the bare rekey carries no EMA state).

Verifies the key count survives the round trip and prints anything it could
not place under a known prefix instead of writing silently.
"""
import argparse
import sys
from collections import OrderedDict

import torch

LIGHTNING_PREFIX = "models.denoiser."
IGNORED_DUPLICATE_PREFIX = "training_models.denoiser."
STANDALONE_PREFIX = "gen3dseg.flow_model."


def extract_bare_denoiser_state(payload, ema_rate: str = None):
    """Return (bare_state_dict, source_description)."""
    if ema_rate is not None:
        if not (isinstance(payload, dict) and "ema" in payload):
            raise ValueError("--ema was given but this checkpoint has no top-level 'ema' key "
                              "(only real lightning training checkpoints carry EMA state).")
        ema_state = payload["ema"].get("state", {})
        if ema_rate not in ema_state:
            raise ValueError(f"EMA rate {ema_rate!r} not found; available: {list(ema_state.keys())}")
        raw = ema_state[ema_rate]
        bare = {k[len(LIGHTNING_PREFIX):]: v for k, v in raw.items() if k.startswith(LIGHTNING_PREFIX)}
        leftover = [k for k in raw if not k.startswith(LIGHTNING_PREFIX)]
        if leftover:
            print(f"[unrekey] WARNING: {len(leftover)} EMA keys did not match prefix "
                  f"'{LIGHTNING_PREFIX}' and were left out: {leftover[:5]}")
        return bare, f"ema state (rate={ema_rate})"

    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        sd = payload["state_dict"]
        bare = {k[len(LIGHTNING_PREFIX):]: v for k, v in sd.items() if k.startswith(LIGHTNING_PREFIX)}
        if bare:
            ignored = [k for k in sd if k.startswith(IGNORED_DUPLICATE_PREFIX)]
            other = [k for k in sd if not k.startswith(LIGHTNING_PREFIX) and not k.startswith(IGNORED_DUPLICATE_PREFIX)]
            if ignored:
                print(f"[unrekey] ignoring {len(ignored)} '{IGNORED_DUPLICATE_PREFIX}*' keys "
                      "(verified register-time duplicate of models.denoiser.*, not separate weights)")
            if other:
                print(f"[unrekey] WARNING: {len(other)} unrelated state_dict keys ignored "
                      f"(not denoiser, not the known duplicate): {other[:5]}")
            return bare, f"lightning state_dict ('{LIGHTNING_PREFIX}*' keys)"

    if isinstance(payload, dict) and isinstance(payload.get("denoiser"), dict):
        return payload["denoiser"], "already-bare {'denoiser': ...} rekey (rekey_full_seg.py output)"

    if isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload, "raw bare state dict (no wrapper)"

    raise ValueError(
        "Could not find a denoiser state dict in this checkpoint. Expected a "
        f"lightning checkpoint with '{LIGHTNING_PREFIX}*' keys under 'state_dict', "
        "a bare {'denoiser': ...} rekey, or a raw state dict."
    )


def unrekey(src: str, dst: str, ema_rate: str = None) -> None:
    payload = torch.load(src, map_location="cpu")
    bare, source_desc = extract_bare_denoiser_state(payload, ema_rate=ema_rate)
    print(f"[unrekey] read {len(bare)} bare keys from {source_desc}")

    out = OrderedDict((STANDALONE_PREFIX + k, v) for k, v in bare.items())
    print(f"[unrekey] {len(out)} keys rekeyed to '{STANDALONE_PREFIX}*' of {len(bare)} in source")
    assert len(out) == len(bare), "key count changed during rekey; refusing to write a partial result"

    torch.save({"state_dict": out}, dst)
    print(f"[unrekey] wrote {dst}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", help="lightning .ckpt, or an already-bare {'denoiser': ...} rekey .pt")
    ap.add_argument("dst", help="output path: eval-compatible standalone-format checkpoint")
    ap.add_argument("--ema", metavar="RATE", default=None,
                     help="pull the EMA shadow at this rate (e.g. 0.999) instead of raw weights; "
                          "only valid when src is a lightning training checkpoint")
    args = ap.parse_args()
    unrekey(args.src, args.dst, ema_rate=args.ema)


if __name__ == "__main__":
    sys.exit(main())
