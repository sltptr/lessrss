<div align="center">
  <img src="https://sltptr.github.io/static/lss/img/logo.png" 
    alt="LSS Logo" style="max-width: 80%; width: 1200px; height: auto;">
</div>

---

### Motivation

Having used a feed reader for the better part of a year, I wanted to see if RSS
could be augmented with a simple recommender system that could highlight the
content I find more interesting. LSS is that solution, just through
click-tracking it learns to predict what items you would read by their titles.

<div align="center">
  <img src="https://sltptr.github.io/static/lss/img/example.png" 
    alt="LSS Logo" style="max-width: 80%; width: 1200px; height: auto;">
  <small>Demo of LSS feeds</small>
</div>

### Features

- Filter with an ensemble of classical TF-IDF regression and transformer-based
  DistilBERT classification.
- Toggle filtering for individual feeds if you want to see all their updates.
- Set weights for each classifier, their weighted softmaxes are added up to
  classify items.
- Inference and most of the service runs inside the container, DistilBERT
  training runs on AWS Sagemaker and requires AWS CLI to be configured on the
  host machine.
- Designed to be self-hosted, I personally run my instance behind a Caddy
  reverse-proxy.

### Basic Setup

1. Copy over the `config` and `crontab` from `config/examples` to `config`, make
   any changes as you'd like.
2. If using DistilBERT, add an `iam_role` field to the config that you'll have
   to create on AWS, also need to have an AWS config at `~/.aws` for said IAM
   role..
3. Run `docker compose up production -d`.

### Updates / Issues

08/12/24 - Has labels for `poor/average/good` ratings that are associated with
red, blue, and star emojis in the generated feeds. TF-IDF classification and
DistilBERT had better F1 scores when they just did binary classification instead
of ternary, looking into a solution that is good for small datasets. Test
coverage needs improvement, alembic mostly unused so migrations are still done
manually, needs documentation for setup and probably an init script. Thinking of
including confidence scores in the generated feeds next to the emojis.
