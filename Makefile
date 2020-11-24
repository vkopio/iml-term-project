run:
	@python -m src

test:
	ENV=test python -m unittest

test-coverage:
	ENV=test coverage run -m unittest
