This file describes how to get data for PushCube in the
Franka Emilia Panda arm. The general idea is we train
a good model (training is very fast) and then evaluate it
a thousand times or so.

First train PPO on franka arm in the push cube environment:

```
python stompy_live/scripts/ppo.py --env_id="New-PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 --eval_freq=10
```

Then run `python stompy_live/scripts/create-data.py`. This
will create 10000 `.json` files, one per episode.

The data consists of images paired with next move taken by the PPO policy.
Unsuccessful episodes are trimmed out.
