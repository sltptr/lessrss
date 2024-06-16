<h1 align="center">LessRSS</h1>
<h3 align="center">Get fewer, better feed items.</h3>

![LessRSS Screenshot w/ FreshRSS](https://sltptr.github.io/static/images/recorss.png)
![Project Status](https://img.shields.io/badge/Status-InDevelopment-red.svg)

## In Progress

- Automated training workflows
- Script to quickly label a dataset of headlines

### Motivation

RSS is a great tool for getting your updates, and having used it for the past year
I wanted to see if there was a way to filter my feeds based on my own preferences for
RSS items. LessRSS is that solution, it's a simple recommendation system where each
item is filtered through an ensemble of machine learning models so that you only get
the RSS items you would be interested in.

### Setup

1. Copy over the `config/default-config.yml` to `config/config.yml`, make changes as you like.
2. Create `.env`, add any necessary envvars.
3. Run `docker compose up`.

### Configuration

Example at `examples/config.yml`.
