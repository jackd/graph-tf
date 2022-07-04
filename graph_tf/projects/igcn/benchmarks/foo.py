import os

import numpy as np
import pyarrow as pa
import tqdm

rows = 1000
cols = 100

path = "/tmp/arrow"
os.makedirs(path, exist_ok=True)
path = os.path.join(path, "data.arrow")

schema = pa.schema([pa.field("nums", pa.float32())])
with pa.OSFile(path, "wb") as sink:
    with pa.ipc.new_file(sink, schema) as writer:
        for row in tqdm.trange(rows):
            batch = pa.record_batch(
                [pa.array(np.random.uniform(size=cols), type=pa.float32())], schema
            )
            writer.write(batch)

src = pa.memory_map(path, "rb")
opened = pa.ipc.open_file(src)
# with pa.memory_map(path, "rb") as source:
#     loaded_array = pa.ipc.open_file(source).read_all()
#     print(type(loaded_array))
#     print(loaded_array.take([1, 3]))
