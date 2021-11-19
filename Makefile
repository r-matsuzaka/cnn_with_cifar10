PACKAGE_DIR=src

format:
	@poetry run isort .
	@poetry run black .

lint:
	@poetry run pylint -d C,R,fixme $(PACKAGE_DIR)

test:
	@poetry run pytest