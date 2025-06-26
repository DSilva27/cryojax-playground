# cryoJAX Playground
A playground to implement things using cryoJAX. Ideas implemented here might be ported to the main repo, or used to build a new downstream package.

# Contributing
Create a fork, implement whatever you want in a new folder and Pull Request. TODO: more specific guidelines.

Make sure your code is compatible with our linting requirements, to do this install `pre-commit` and `ruff`

```bash
pip install pre-commit ruff
```

Then, install the pre-commit hooks while in the repo folder

```bash
cd path/to/cryojax-playground
pre-commit install
```

Pre-commit should then check automatically your code before a commit, otherwise you can run:
```
pre-commit run --all-files
```
