# Paper

The paper reports the pilot experiment of SpherIDiff in the NeurIPS preprint format.

- `main.tex`: the source of the paper, with its bibliography.
- `neurips_2026.sty`: the NeurIPS style file.
- `figures/`: the figures of the paper.
- `make_figures.py`: produces every figure from the result files in `results/` and the local data caches. It is run with `python paper/make_figures.py` from the repository root.

The paper is compiled with `pdflatex main.tex` inside this folder.
