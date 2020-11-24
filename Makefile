run:
	@python -m src

test:
	ENV=test python -m unittest

test-coverage:
	ENV=test coverage run -m unittest

dev-install:
	virtualenv --python=/usr/bin/python3.7 venv
	. venv/bin/activate
	pipenv install --dev
