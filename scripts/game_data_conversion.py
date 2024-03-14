import re
import time
import uuid

import jsonlines
import pandas as pd

start = time.time()
for idx, chunk in enumerate(pd.read_csv("data/lichess_100mb/lichess_100mb.csv", chunksize=1000)):
    print(idx)
    with jsonlines.open("data/lichess_100mb/lichess_100mb.jsonl", mode="a") as writer:
        for index, row in chunk.iterrows():
            moves = row["transcript"]
            # replacing all periods with periods and spaces
            moves = re.sub(r"\.", ". ", moves)
            game = {"gameid": str(uuid.uuid4()), "moves": moves}
            writer.write(game)
print(f"Time Taken: {time.time() - start} seconds", flush=True)
