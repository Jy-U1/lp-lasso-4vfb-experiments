.PHONY: install test smoke verify assets paper package

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

smoke:
	python run_all.py --profile smoke --experiments all

verify:
	python scripts/verify_manuscript_results.py

assets:
	python scripts/build_paper_assets.py

paper:
	python run_all.py --profile paper --experiments all

package:
	python scripts/package_release.py
