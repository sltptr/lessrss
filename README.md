<h1 align="center">LessRSS</h1>
<h3 align="center">Get fewer, better feed items.</h3>

![LessRSS Screenshot w/ FreshRSS](https://sltptr.github.io/static/images/recorss.png)

### Motivation

RSS is a great tool for getting your internet updates, and having used it for the past year
I wanted to see if there was a way to filter my feeds based on data collected with click tracking.
LessRSS is that solution, it's a simple recommendation system that filters items through
classifiers before publishing them to your feed.

### Setup

1. Copy over the `config/default-config.yml` to `config/config.yml`, make changes as you like.
2. Create `.env`, add necessary envvars.
3. Run `docker compose up`.

### In Progress

- Automated training workflows
- Cold-start script to quickly label news headlines
