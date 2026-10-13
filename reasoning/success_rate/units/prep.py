"""Refresh predictor data from s3 and pickle NFL's rated seasons once."""
import asyncio
import pickle
import sys
from pathlib import Path

from endgame_aws import Config

import cassandra
from cassandra.batch.artifacts import download_predictor_data
from cassandra.save_predictions import read_rated_seasons

OUT = Path.home() / ".cassandra/success_rate/units/nfl_seasons.pkl"

async def main():
    print("cassandra from", cassandra.__file__)
    bucket = Config.init_from_file().bucket
    paths = await download_predictor_data(bucket)
    print("downloaded", len(paths), [p.name for p in paths if "nfl" in p.name])
    seasons = await read_rated_seasons("nfl", bucket)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(pickle.dumps(seasons))
    print(len(seasons), "seasons", seasons[0].year, "-", seasons[-1].year)

asyncio.run(main())
