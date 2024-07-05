# Stompy, Live!

Like Twitch Plays Pokemon for Humanoid Robots.

Local Installation:
```bash
git clone https://github.com/kscalelabs/stompy-live
&& cd stompy-live
&& conda create -y -n "stompylive" python=3.11
&& conda activate stompylive
&& pip install -e .
```
# From ManiSkill2 Docs for downloading assets:
Some environments require downloading assets. You can download all the assets by  ` python -m mani_skill.utils.download_asset all ` or download task-specific assets by ` python -m mani_skill.utils.download_asset ${ENV_ID} ` . The assets will be downloaded to ` ./data/ ` by default, and you can also use the environment variable ` MS2_ASSET_DIR `to specify this destination.

# Tests:

Test Maniskill:
```bash
python -m tests.maniskilltest
python -m tests.parallelmaniskilltest
python -m tests.test_env
```

## Fine-tuning VLA
Modify data_scripts.datasets to load data properly, and run data_scripts.finetune

## Environment variables

- `STOMPYLIVE_TOKEN`: get from https://twitchapps.com/tmi/
