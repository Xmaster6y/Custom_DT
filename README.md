# Custom DT

Customised version of Decision Transformers.

## Local Development


The project dependencies are managed using `poetry`, see their installation [guide](https://python-poetry.org/docs/). For even more stability, I recommend using `pyenv` or python `3.9.18`.

Additionally, to make your life easier, install `make` to use the shortcut commands.

## Base Install

To install the dependencies:

```bash
poetry install
```

## Dev Install

To install the dependencies:

```bash
poetry install --with dev
```

Before committing, install `pre-commit`:

```bash
pre-commit install
```

To run the checks (`pre-commit` checks):

```bash
make checks
```

## Data

In order to train a model you need to download the datasets.

- [BlueSunflower/chess_games_base](https://huggingface.co/datasets/BlueSunflower/chess_games_base)
