import wandb

api = wandb.Api()
for r in api.runs("ngochale2012-hust/car-damage-mlops", filters={"jobType": "train"})[:10]:
    print(
        f"{r.name:25} lr={r.config['train']['lr']:<8} best={r.summary.get('val/acc')} "
        f"data={r.config['data_md5'][:8]} exp={r.config.get('dvc_exp')}"
    )
m = api.artifact("wandb-registry-model/car-damage-classifier:staging")
print("staging:", m.version, m.metadata)
print("lineage:", [a.name for a in m.logged_by().used_artifacts()])
