<div align="center">
  <img src="https://sltptr.github.io/static/images/LSSLogo.png?" 
    alt="LSS Logo" style="max-width: 80%; width: 1200px; height: auto;">
</div>

<p align="center">(but still simple enough)</p>

### Motivation

RSS is great, but there's always opportunities for improvement. Having used a
feed reader for the better part of a year, I wanted to see if RSS could be
augmented with a simple recommender system that highlights the more interesting
content.

LSS is that solution, through simple click-tracking it learns to predict if you
would read new entries by their titles. Effectively, this means LSS is a simple
content-based filter that you can wrap around your feeds.

### Features

- Filter with a combination of TF-IDF Logistic Regression and DistilBERT
  classification.
- Toggle show all results from a feed, regardless of predictions.
- Set default and specific weighted predictions for each feed.

### Setup

1. Copy over the examples from `config/examples` to `config`, make any changes
   as you'd like.
2. If using DistilBERT, create `.env` and add the necessary envvar `IAM_ROLE`,
   requires being logged into the AWS CLI at `~/.aws`.
3. Run `docker compose up -d`.
