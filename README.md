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
Tests:

Test Maniskill:
```bash
python -m tests.maniskilltest
python -m tests.parallelmaniskilltest
python -m tests.test_env
```

## Environment variables

- `STOMPYLIVE_TOKEN`: get from https://twitchapps.com/tmi/
