<div style="text-align: center;">
  <img src="https://sltptr.github.io/static/images/LessRssLogo.png" 
    alt="Less RSS Logo" style="max-width: 60%; width: 600px; height: auto;">
</div>

<h3 align="center">Get fewer, better feed items.</h3>

### Motivation

RSS is a great tool for getting your internet updates, and having used it for
the past year I wanted to see if there was a way to filter my feeds based on
data collected with click tracking. LessRSS is that solution, it's a simple
recommendation system that filters items through classifiers before publishing
them to your feed.

### Features

- Filter with a combination of TF-IDF Logistic Regression and DistilBERT
  classification.
- Toggle show all results from a feed, regardless of predictions.
- Set default and specific weighted predictions for each feed.

### Setup

1. Copy over the `config/default-config.yml` to `config/config.yml`, make
   changes as you'd like.
2. Create `.env`, add necessary envvars `SQLALCHEMY_URL` and `IAM_ROLE` (only if
   using DistilBERT).
3. Run `docker compose up --build -d`.
