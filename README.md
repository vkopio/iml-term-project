# iml-term-project

The term project for Introduction to Machine Learning course.

## Install

This project uses Python `3.8`. To install the dependencies you need `pipenv` which can be installed with

    pip install pipenv

Then install the dependencies with

    pipenv --python 3.8
    pipenv shell
    pipenv install --dev


## Run

The project can be run by first activating the `pipenv` virtualenv 

    pipenv shell

After the virtualenv is activated simply use the commands from `Makefile`

    make run

## Test

Test can be run with

    make test

## Notebook setup with virtualenv

To get Jupyter lab to use the virtualenv you need to register it first by running the following:

    pipenv shell
    pip install ipykernel
    python -m ipykernel install --user --name=iml-term-project
