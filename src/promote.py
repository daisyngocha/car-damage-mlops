"""Link 1 version model vào W&B Registry kèm alias. Dùng: promote.py <entity>/<project>/car-damage-model:vN staging"""

import sys

import wandb

src, alias = sys.argv[1], sys.argv[2]
art = wandb.Api().artifact(src)
art.link("wandb-registry-model/car-damage-classifier", aliases=[alias])
print(f"linked {art.name} → car-damage-classifier: {alias} (val_acc={art.metadata.get('val_acc')})")
