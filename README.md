# aidev-pr-analysis

This repository contains the c# AIDev PR Interaction Analysis

This repository contains the codebase used to conduct the analyses in our paper and to support replication of our results.

## Dataset

We base our analysis on the AIDev v1.0 dataset introduced by Li et al. (2025).  
Due to size and licensing constraints, the dataset is not included in this repository.

Please follow the official AIDev instructions to obtain the dataset.

After downloading, place the data under:

data/raw/

## Project Structure

scripts/ # data extraction and preprocessing
analysis/ # RQ-specific analysis scripts
notebooks/ # exploratory analysis
results/ # generated tables and figures
data/ # raw and processed data (not tracked in git)

## Reproducing the Results

High-level steps to reproduce the analysis:

1. Download the AIDev dataset and place it under `data/raw/`
2. Run preprocessing scripts in `scripts/`
3. Run analysis scripts in `analysis/` for each research question

## Notes

This repository does not include the raw dataset. Please refer to the original AIDev publicatdebase used to answer the research questions in this study and to support replication of our results.
