This file describes how to get data for PushCube in the
Franka Emilia Panda arm. The general idea is we train
a good model (training is very fast) and then evaluate it
a thousand times or so.

First run PPO on franka arm in the push cube environment:

```
python stompy_live/scripts/ppo.py --env_id="New-PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 --eval_freq=10
```

Then run `python stompy_live/scripts/create-data.py`. This
will create 1000 `.h5` files, one per episode.