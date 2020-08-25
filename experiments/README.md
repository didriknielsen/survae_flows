# Experiments

Code to reproduce the experiments in Sec. 4 of [SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows](https://arxiv.org/abs/2007.02731).  
We perform 3 sets of experiments:
1. **Toy Experiments:** We show how absolute value surjections may be used to enforce symmetries on the learned density.
1. **Point Cloud Experiments:** We show how sorting surjections and stochastic permutation layers may be used to enforce permutation invariance on the learned density.
1. **Image Experiments:** We compare max pooling surjections and tensor slicing surjections for downsampling in a flow model for image data.

More details are given in each sub-folder.
