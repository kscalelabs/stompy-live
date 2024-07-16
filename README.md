# Stompy, Live!

Like Twitch Plays Pokemon for Humanoid Robots.

Local Installation:

```bash
git clone https://github.com/kscalelabs/stompy-live
&& cd stompy-live
&& conda create -y -n "stompylive" python=3.11
&& conda activate stompylive
&& make install
```

# From ManiSkill2 Docs for downloading assets:

Read if you want to manually install the ManiSkill assets rather than using `make install`.

Some environments require downloading assets. You can download all the assets by `python -m mani_skill.utils.download_asset all` or download task-specific assets by ` python -m mani_skill.utils.download_asset ENV_ID` . The assets will be downloaded to `~/.maniskill/data/` by default, and you can also use the environment variable `MS_ASSET_DIR`to specify this destination.

You will want to, at a minimum, download

- ReplicaCAD
- AI2THOR

by replacing `ENV_ID` with the given datasets in the previously given command.

# Tests:

Test Maniskill:

```bash
python -m tests.maniskilltest
python -m tests.parallelmaniskilltest
python -m tests.test_env
```

## Environment variables

- `STOMPYLIVE_TOKEN`: get from https://twitchapps.com/tmi/
