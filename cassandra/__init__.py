"""Cassandra: rating models, and the release artifact they publish.

The package splits the same way the dependencies do. `cassandra.serving`,
`cassandra.predictor` and `cassandra.prob_to_margin` are the release-reading
half, and import nothing that talks to s3 -- which is what lets the webapp
install cassandra without the `fit` group. `cassandra.save_predictions` and
`cassandra.odds` read the bucket, and `cassandra.model_eval` drives them.

Nothing is re-exported here, deliberately. `cassandra` is a package, so
`import cassandra.serving` runs this file first, and a `from .model_eval
import ...` at the top of it would pull `endgame_aws` -- aiobotocore, and
pyarrow behind it -- into a consumer that only wanted to read a saved
calibration. Every name lives in the module that owns it.
"""
