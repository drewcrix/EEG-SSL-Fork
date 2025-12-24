# Integrating Neurological Priors Into Self Supervised Learning for Electroencephalography
This repository contains the source code for the article:
Integrating Neurological Priors Into Self Supervised Learning for Electroencephalography.

This project is an extension of the BENDR source and paper. The goal of this project is to investigate the viability of applying prior information from neuroscience to improve an existing framework for performing self supervised learing on EEG data. Where BENDR directly maps SSL methodologies from NLP, our approach modifies it so that it is more concretely grounded in the electrophysiological and cognitive priors of the data. We do this in two ways, firstly with a more robust reduction methodology that maps arbitrary sensor counts and configurations and secondly with a contrastive learning method more strongly rooted in cognitive neuroscience. 

BENDR downsamples multichannel EEG data into a single channel for training by using 1D convolutions, which constrains it to the channel count that they truncate to. Furthermore, this approach is not robust to different sensor layouts, which can be the case depending on the equipement used for the reading. This is why we use a graph net informed with the physical information of the layout for sequence reduction insted. This is an existing technique from [EEG decoding for datasets with heterogeneous electrode configurations using transfer learning graph neural networks, 2023](https://iopscience.iop.org/article/10.1088/1741-2552/ad09ff/meta)

BENDR does contrastive learning like wav2vec, trying to unmix permutations of eeg time series sequences. This approach is very sensible in the context of language processing as spoken sentences are known to be based on highly regular grammars with relatively few valid permuations relative to invalid ones along with having have definate start and ends. In the context of EEG data, none of these priors can be assumed to hold. We instead train by clustering segments by task-switching, through models validated by existing neuroscience work. The logic being that segmenting every time task switching occurs, we can track different thought processes, mental states, and actions, which is highly valuble, and rooted in validated principles.

Below is the instructions for running it (currently copied from BENDR, pending rewrite)


## BENDR 
*BErt-like Neurophysiological Data Representation*

This repository contains the source code for reproducing, or extending the BERT-like self-supervision pre-training for EEG data from the article:

[BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data](https://arxiv.org/pdf/2101.12037.pdf)

To run these scripts, you will need to use the [DN3](https://dn3.readthedocs.io/en/latest/) project. We will try to keep this updated so that it works with the latest DN3 release. If you are just looking for the BENDR model, and don't need to reproduce the article results *per se*, BENDR will be (or maybe already is if I forgot to update it here) integrated into DN3, in which case I would start there.

Currently, we recommend version [0.2](https://github.com/SPOClab-ca/dn3/tree/v0.2-alpha). Feel free to open an issue if you are having any trouble.

More extensive instructions are upcoming, but in essence you will need to either:

    a)  Download the TUEG dataset and pre-train new encoder and contextualizer weights, _or_
    b)  Use the [pre-trained model weights](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha)
        
Once you have a pre-trained model:

    1) Add the paths of the pre-trained weights to configs/downstream.yml
    2) Edit paths to local copies of your datasets in configs/downstream_datasets.yml
    3) Run downstream.sh
