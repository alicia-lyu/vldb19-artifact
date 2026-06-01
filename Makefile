.PHONY: paper-ready plots clean

RESULTS ?= results
PYTHON  ?= python3

# Stage 1: container sweep. `docker_run.sh` is authored separately (it
# pulls the pre-built LeanStore + DBToaster images from GHCR, runs the
# sweep matrix documented in README.md, and writes a paper-data-shaped
# tree at $(RESULTS)/<tag>/{summary,manifest.yaml}). The stamp file
# marks completion so re-runs of `make plots` skip the sweep.
$(RESULTS)/.stamp:
	./docker_run.sh --results $(RESULTS)
	@touch $@

# Stage 2: pure plotting. Idempotent over an existing $(RESULTS) tree.
plots: $(RESULTS)/.stamp
	$(PYTHON) main.py --results $(RESULTS) --out paper-ready/

paper-ready: plots

clean:
	rm -rf paper-ready/*.pdf paper-ready/*.png paper-ready/*.csv $(RESULTS)
