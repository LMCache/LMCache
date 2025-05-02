# Contributing

👍🎉 First off, thank you for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## What Should I Know Before I Get Started?

### Code of Conduct

This project adheres to the [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

### How Do I Start Contributing?

The below workflow is designed to help you begin your first contribution journey. It will guide you through creating and picking up issues, working through them, having your work reviewed, and then merging.

Help on open source projects is always welcome and there is always something that can be improved. For example, documentation (like the text you are reading now) can always use improvement, code can always be clarified, variables or functions can always be renamed or commented on, and there is always a need for more test coverage. If you see something that you think should be fixed, take ownership! Here is how you get started:

## How Can I Contribute?

When contributing, it's useful to start by looking at [issues](https://github.com/LMCache/LMCache/issues). After picking up an issue, writing code, or updating a document, make a pull request and your work will be reviewed and merged. If you're adding a new feature or find a bug, it's best to [write an issue](https://github.com/LMCache/LMCache/issues/new) first to discuss it with maintainers.

To contribute to this repo, you'll use the Fork and Pull model common in many open source repositories. For details on this process, check out [The GitHub Workflow
Guide](https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md)
from Kubernetes.

When your contribution is ready, you can create a pull request. Pull requests are often referred to as "PR". In general, we follow the standard [GitHub pull request](https://help.github.com/en/articles/about-pull-requests) process. Follow the template to provide details about your pull request to the maintainers. It's best to break your contribution into smaller PRs with incremental changes, and include a good description of the changes. We require new unit tests to be contributed with any new functionality added.

Before sending pull requests, make sure your changes pass formatting, linting and unit tests. These checks will run with the pull request builds. Alternatively, you can run the checks manually on your local machine [as specified below](#development).

#### Code Review

Once you've [created a pull request](#how-can-i-contribute), maintainers will review your code and may make suggestions to fix before merging. It will be easier for your pull request to receive reviews if you consider the criteria the reviewers follow while working. Remember to:

- Run tests locally and ensure they pass
- Follow the project coding conventions
- Write detailed commit messages
- Break large changes into a logical series of smaller patches, which are easy to understand individually and combine to solve a broader issue

Maintainers will perform "squash and merge" actions on PRs in this repo, so it doesn't matter how many commits your PR has, as they will end up being a single commit after merging.

## Development

### Set up your dev environment

The following tools are required:

- [git](https://git-scm.com)
- [python](https://www.python.org) (v3.10+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

The first step is to install the necessary Python packages required for development. The commands to do this are as follows:

```shell
pip install -e .
pip install -r requirements-lint.txt
pip install -r requirements-test.txt
pip install -r requirements.txt
```

Before pushing changes to GitHub, you need to run the tests and coding style as shown below.

### Unit tests

When making changes, run the tests before pushing the changes. Running unit tests ensures your contributions do not break exiting code. We use [pytest](https://docs.pytest.org/) framework to run unit tests. The framework is setup to run all run all test_*.py or *_test.py in the [tests](./tests) directory.

Running unit tests is as simple as:

```sh
pytest
```

By default, all tests found within the tests directory are run. However, specific unit tests can run by passing filenames, classes and/or methods to `pytest`. The following example invokes a single test method `test_lm_connector` that is declared in the `tests/test_connector.py` file:

```shell
pytest tests/test_connector.py::test_lm_connector
```

### Coding style

LMCache follows the Python [pep8](https://peps.python.org/pep-0008/) coding style. We use [isort](https://pycqa.github.io/isort/), [mypy](https://github.com/python/mypy), [yapf](https://github.com/google/yapf), and [Ruff](https://docs.astral.sh/ruff/) for Python linting and formatting.

We also use [codespell](https://github.com/codespell-project/codespell) for spell checking and [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for C++ code formatting.

You can invoke all linting and formatting with the following command:

```sh
bash format.sh
```

## Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issues:

- Issues with the [`good first issue` label](https://github.com/LMCache/LMCache/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - these should only require a few lines of code and are good targets if you're just starting contributing.
- Issues with the [`help wanted` label](https://github.com/LMCache/LMCache/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) - these range from simple to more complex, but are generally things we want but can't get to in a short time frame.
