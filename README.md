# musicmri - Reconstructing Auditory Stimuli via Feature-based Clustering

## Abstract

## Introduction

## Methods

Initially used Italian features with a CNN; features too rare in real music anyway
Pivoted to PCA->UMAP for arbitrary features
Testing Italian features on UMAP, failure to accurately classify
    "Italian features are a dead end. The features just... don't audibly match, and it still runs into the downstream problem that they don't cleanly appear in the stimuli or modern music in the first place. So my conclusion is that keeping with the dataset-specific features is the way to go"

Relative validity was a failure (kind of). For granular feature extraction, it insisted upon a larger min_cluster_size. The best route forward is to keep mcs_max at a low ceiling
    Granular feature extraction helps to get minor variations in spectral shapes; marginally different drum riffs for ex

Training set needs to be contained to avoid cluster formation dominated by disparate pieces of the corpus

## Results

## Discussion

## Acknowledgements

## Data Availability

## Code Availability

## Footnotes

## References

## Associated Data

### Data Availability Statement