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
  <small>Example of LSS feeds in a feed reader</small>
</div>

### Features

- See which RSS items you would read (&#11088;), might read (&#128309;), and
  probably wouldn't read (&#128308;). The items you probably wouldn't read only
  pass through the filter if you toggle `filter: false` in your `config.yml`.
- HTTP GET the URLs of all your filtered feeds at `/feeds`, paths are auto
  generated to keep the `config.yml` minimal.
- Clicking an item's link updates the title as average (&#128309;), and links to
  update the item as either good (&#11088;) or poor (&#128308;) are added to the
  description of the item.
- Filter with an ensemble of classical TF-IDF regression and transformer-based
  DistilBERT classification.
- Toggle filtering for individual feeds if you want to see all their updates.
- Set weights for each classifier, their weighted softmaxes are added up to
  classify items.
- Inference and most of the service runs inside the container, DistilBERT
  training runs on AWS Sagemaker and requires the AWS CLI to be configured on
  the host machine.

### Basic Setup

0. Recommended to be run behind a reverse-proxy with authentication, default
   port is 5000.
1. Copy over the `config.yml` and `crontab` from `config/examples` to `config`,
   make any changes as you'd like.
2. If using DistilBERT, add an `iam_role` field to the config that you'll have
   to create on AWS, also need to have an AWS config at `~/.aws` for said IAM
   role.
3. Starting out, I recommend to set `cold_start: true` at the top of your
   `config.yml` until you have enough data for the training cron jobs to work.
4. Run `docker compose up production -d`

### Updates / Issues

08/12/24 - Has labels for `poor/average/good` ratings that are associated with
red, blue, and star emojis in the generated feeds. TF-IDF classification and
DistilBERT had better F1 scores when they just did binary classification instead
of ternary, looking into a solution that is good for small datasets. Test
coverage needs improvement, alembic mostly unused so migrations are still done
manually, needs documentation for setup and probably an init script. Thinking of
including confidence scores in the generated feeds next to the emojis.
