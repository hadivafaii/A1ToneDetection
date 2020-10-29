Content
=======================

This update includes 3 sections:
1. Data processing
2. Classification Analysis
3. Linear Discriminant Analysis

## What's new:

LDA with different dimensionalities
    
    Improvement in best regularization hyperparameter and best
    timepoint selection.  In this version I'm smoothing the scores
    by calculating a running average using a width 5 kernel.
    This way, selected timepoints end up more real.  A drawback of
    doing this was a ~0.05 drop in performance on average.
   
B) Excitations & Suppression,
    
    Separately visualizing cells that are on average excited (EXC)
    or suppresed (SUP) during a trial.  The tag was determined by first
    averaging DFFs for a given cell during trials of the same type. Then,
    calling a cell EXC if the maximum absolute value of this
    average was positive, and SUP if it was negative.
    Moreover, distribution of EXC/SUP cells are plotted for all
    tial types (more discussion in the book).

C) Shuffled LDA,
    
    As anticipated, shuffling the training set labels completely destorys
    the coherence we observe for the true dataset, proving those diverging
    trajectories are real.
    
D) LDA with different number of dimensions,

    summary: performance does not change that much, trajectories are
    more separable for larger dimensions (obviously)

E) Calcium temporal convolution: *Calium Tent*,

    Discussed the temporal convolution properties of Ca traces
    with Nik.  In neuroscience people typically use genetically encoded
    calcium indicators rather than chemical, which is what Nik also uses.
    This choice has many benefits such as a more-or-less identical
    convolution filter for cells.  The decay time is roughly a few
    hundred milliseconds; however, it is still not entirely clear how can
    we use this information in our analysis.  For now, we can keep this
    piece of information in mind when interpreting the results.
    Hopefully, later we can build a more advanced model that
    automatically learns this dynamcis and separates it from other
    sources of variation.

Lastly, Nik suggested a paper about this:
[Ultrasensitive fluorescent proteins for imaging neuronal activity](https://www.nature.com/articles/nature12354).
I saw this paper before, but haven't read it in detail yet.
Will try to learn the main points that will be relevant for us.