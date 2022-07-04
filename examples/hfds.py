"""Huggingface datasets playground."""


import datasets
import numpy as np

logger = datasets.logging.get_logger(__name__)


_CITATION = """."""

_DESCRIPTION = """."""

n = 170_000
c = 40
# l = 100_000
k = 10_000


class Dummy(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "linear": datasets.features.Sequence(
                        datasets.Value("float32"), length=c
                    ),
                    "quad": datasets.features.Sequence(
                        datasets.Value("float32"), length=k
                    ),
                    # "x": [datasets.Value("float32")],
                    # "y": [datasets.Value("float32")],
                }
            ),
            supervised_keys=None,
            homepage=".",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"count": n, "seed": 0},
            ),
        ]

    def _generate_examples(self, count: int, seed: int):
        """This function returns the examples in the raw (text) form."""
        rng = np.random.default_rng(seed)
        for i in range(count):
            yield i, {
                "linear": rng.normal(size=(c,)).astype(np.float32),
                "quad": rng.normal(size=(k,)).astype(np.float32),
            }


if __name__ == "__main__":
    import tqdm

    builder = Dummy()
    # builder.download_and_prepare()

    dataset = builder.as_dataset(datasets.splits.Split.TRAIN, run_post_process=False)

    def get_element(
        dataset: datasets.Dataset, batch_size: int, rng: np.random.Generator
    ):
        indices = rng.choice(n, batch_size)
        out = dataset[indices]
        lin = out["linear"]
        quad = out["quad"]
        # lin = np.array(out["linear"])
        # quad = np.array(out["quad"])
        return lin, quad

    steps_per_epoch = 1000
    batch_size = 1_000
    rng = np.random.default_rng(0)

    for _ in tqdm.trange(steps_per_epoch):
        get_element(dataset, batch_size, rng)
