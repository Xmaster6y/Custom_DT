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

To run the tests (`pytest` tests):

```bash
make tests
```

## Data

See the [data](./data/) folder.

## Weights

See the [weights](./weights/) folder.

## Logs

See the [logging](./logging/) folder.

## To Use Stockfish

1. Download the stockfish source code from this [link](https://stockfishchess.org/download/) and save the zip file in the top level of this repository
2. Extract the files from the zip file
3. Rename the top folder to `stockfish-source`
