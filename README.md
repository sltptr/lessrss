<div align="center">
  <img src="https://sltptr.github.io/static/images/LSSLogo.png?" 
    alt="LSS Logo" style="max-width: 80%; width: 1200px; height: auto;">
</div>

<p align="center">(but still simple enough)</p>

### Motivation

RSS is great, but there's always opportunities for improvement. Having used a feed
reader for the better part of a year, I wanted to see if RSS is could be augmented with
a simple recommender system so I could filter out new entries I'd rather not read.

LSS is that solution, through click-tracking it learns to predict if
you would or would not read new entries by their titles. Effectively, LSS is a simple
content-based filter for your RSS feeds.

### Features

- Filter with a combination of TF-IDF Logistic Regression and DistilBERT
  classification.
- Toggle show all results from a feed, regardless of predictions.
- Set default and specific weighted predictions for each feed.

### Setup

1. Copy over the `config/default-config.yml` to `config/config.yml`, make
   changes as you'd like.
2. Create `.env`, add necessary envvars `SQLALCHEMY_URL` and `IAM_ROLE` (only if
   using DistilBERT, requires being logged into the AWS CLI).
3. Run `docker compose up -d`.
