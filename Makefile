.PHONY: paper-ready plots smoke clean

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

# Fast end-to-end validation at SF=15 (a few minutes): run the smoke sweep and
# copy the produced figures + macros into paper-ready/. Always runs (no stamp)
# — it is a one-off sanity check, not the cached multi-hour full sweep. Use this
# to confirm the host + image work before committing to `make paper-ready`.
smoke:
	./docker_run.sh --smoke --results $(RESULTS)
	$(PYTHON) main.py --results $(RESULTS) --out paper-ready/

clean:
	rm -rf paper-ready/*.pdf paper-ready/*.png paper-ready/*.csv $(RESULTS)
