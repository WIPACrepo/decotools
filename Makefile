tests:
	py.test -v decotools

deploy-docs:
	cd docs; mkdocs gh-deploy --clean; cd -;
