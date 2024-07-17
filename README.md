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

## Downloading Stompy URDF

> [!WARNING]
> This is temporary! Soon we will have the Stompy URDF on media.kscale.dev and automate its installation in a Make rule.

Download

https://drive.google.com/drive/folders/1dNL8i4sfu5N6ojUMOb9YDcCMRc3jRXg2

and unzip in stompy_live/assets

Then replace "fused" with "meshes/fused" in the main urdf file (robot_7dof_arm_merged_simplfied.urdf).

## From ManiSkill2 Docs for downloading assets:

Some environments require downloading assets. You can download all the assets by  ` python -m mani_skill.utils.download_asset all ` or download task-specific assets by ` python -m mani_skill.utils.download_asset ENV_ID` . The assets will be downloaded to ` ./data/ ` by default, and you can also use the environment variable ` MS2_ASSET_DIR `to specify this destination.

You will want to, at a minimum, download

- ycq
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
