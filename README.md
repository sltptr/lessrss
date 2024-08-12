<div align="center">
  <img src="https://sltptr.github.io/static/images/LSSLogo.png?" 
    alt="LSS Logo" style="max-width: 80%; width: 1200px; height: auto;">
</div>

### Updates / Issues

08/12/24 - Test coverage needs improvement, alembic mostly unused, needs
documentation for initial setup and probably an init script

### Motivation

Having used a feed reader for the better part of a year, I wanted to see if RSS
could be augmented with a simple recommender system which highlights the content
I'd find more interesting. LSS is that solution, just through click-tracking it
learns to predict if you would read new RSS items by their titles.

### Features

- Filter with a combination of classical TF-IDF regression and transformer-based
  DistilBERT classification.
- Toggle filtering for individual feeds if you want to see all items.
- Set weights for each classifier, their weighted softmaxes are added up to
  classify items.

### Setup

1. Copy over the examples from `config/examples` to `config`, make any changes
   as you'd like.
2. If using DistilBERT, create `.env` and add the necessary envvar `IAM_ROLE`,
   requires being logged into the AWS CLI at `~/.aws`.
3. Run `docker compose up production -d`.
