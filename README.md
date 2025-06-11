# Clementi Palazzo (2016) Replication

This repository replicates the main results of [Clementi and Pallazo (2016)](https://www.aeaweb.org/articles?id=10.1257/mac.20150017) paper. The model is solved using the [sequnece-space-Jacobian-toolkit](https://github.com/shade-econ/sequence-jacobian) based on [Auclert et al., Econometrica (2021)](https://www.econometricsociety.org/publications/econometrica/2021/09/01/using-sequence-space-jacobian-solve-and-estimate-heterogeneous) paper.

## Repository Structure
- [main_nb](main_nb.ipynb): Jupyter notebook that runs the full replication pipeline, including:
  - calibration
  - the steady-state computations
  - transition dynamics
- [blocks_firms.py](blocks_firms.py): Python script containing the model blocks used in the SSJ framework
- [Notes.pdf](Notes.pdf): Provides a detailed overview of the:
  - underlying model
  - computational method
  - replicated results and comparisons

## References:
1. Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub, “Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models,” Econometrica, 2021, 89 (5), 2375–2408
2. Clementi, Gian Luca and Berardino Palazzo, “Entry, Exit, Firm Dynamics, and Aggregate Fluctuations,” American Economic Journal: Macroeconomics, July 2016, 8 (3), 1–41
