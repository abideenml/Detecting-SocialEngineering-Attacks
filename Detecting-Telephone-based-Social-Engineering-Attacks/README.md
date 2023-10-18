## Detecting Telephone-based Social Engineering Attacks Using Document Vectorization and Classification of Scam Signatures :phone: = :cl:
This repo contains the Scam Signature detection dataset and collection of various Vectorization, Clustering and Classification techniques to effectively detect telephone-based social engineering attacks. This repo uses a social engineering detection approach called the Anti-Social Engineering Tool (ASsET), which detects attacks based on the semantic content of the conversation. (:link: [Ali et al.](https://dl.acm.org/doi/pdf/10.1145/3445970.3451152)). <br/>
It's aimed at exploring various ways to detect **Scam Signatures** about transformers. <br/>

## Table of Contents
  * [What are Telephone-based Social Engineering attacks?](#what-are-telephone-based-social-engineering-attacks)
  * [Understanding Document Vectorization](#understanding-vectorization)
  * [Clustering Techniques](#clustering-techniques)
  * [Classification Models](#classification-techniques)
  * [Setup](#setup)
  * [Usage](#usage)
  * [Hardware requirements](#hardware-requirements)

## What-are-Telephone-based-Social-Engineering-attacks

Telephone-based social engineering attacks, also known as phone scams, are a form of cyber attack where malicious actors use the phone as a medium to manipulate individuals or organizations into revealing sensitive information, performing actions, or providing financial gains. These attacks rely on the art of persuasion, psychological manipulation, and impersonation to deceive and exploit victims. 

Each scam type is identified by a set of speech acts that are collectively referred to as a **Scam Signature**. We can define a scam signature as a set of utterances that perform speech acts that are collectively unique to a class of social engineering attacks. These utterances are the key points, fulfilling the goal of the scammer
for that attack. A scam signature uniquely identifies a class of social engineering attacks in the same way that a malware signature uniquely identifies a class of malware. I will use a social engineering detection approach called the **Anti-Social Engineering Tool (ASsET)**, which detects attacks based on the semantic content of the conversation.

<p align="center">
<img src="data/readme_pics/scam-signature.PNG" width="700"/>
</p>

## Understanding Vectorization

In this repo, I have used two different methods for vectorization of scam signatures. First one is **Doc2Vec** and second is **Universal Sentence Encoder**.

**Doc2Vec** is a neural network-based approach that learns the distributed representation of documents. It is an unsupervised learning technique that maps each document to a fixed-length vector in a high-dimensional space. The vectors are learned in such a way that similar documents are mapped to nearby points in the vector space. This enables us to compare documents based on their vector representation and perform tasks such as document classification, clustering, and similarity analysis.

There are two main variants of the Doc2Vec approach: 

1) Distributed Memory (DM)
2) Distributed Bag of Words (DBOW)


➡️ Distributed Memory (DM)
Distributed Memory is a variant of the Doc2Vec model, which is an extension of the popular Word2Vec model. The basic idea behind Distributed Memory is to learn a fixed-length vector representation for each piece of text data (such as a sentence, paragraph, or document) by taking into account the context in which it appears. In the DM architecture, the neural network takes two types of inputs:  the context words and a unique document ID. The context words are used to predict a target word, and the document ID is used to capture the overall meaning of the document. The network has two main components:  the projection layer and the output layer.
The projection layer is responsible for creating the word vectors and document vectors. For each word in the input sequence, a unique word vector is created, and for each document, a unique document vector is created. These vectors are learned through the training process by optimizing a loss function that minimizes the difference between the predicted word and the actual target word. The output neural network takes the distributed representation of the context and predicts the target word.

<p align="center">
<img src="data/readme_pics/DM.PNG" width="400"/>
</p>


➡️ Distributed Bag of Words (DBOW): DBOW is a simpler version of the Doc2Vec algorithm that focuses on understanding how words are distributed in a text, rather than their meaning. This architecture is preferred when the goal is to analyze the structure of the text, rather than its content. In the DBOW architecture, a unique vector representation is assigned to each document in the corpus, but there are no separate word vectors.  Instead, the algorithm takes in a document and learns to predict the probability of each word in the document given only the document vector. The model does not take into account the order of the words in the document, treating the document as a collection or “bag ” of words. This makes the DBOW architecture faster to train than DM, but potentially less powerful in capturing the meaning of the documents.

<p align="center">
<img src="data/readme_pics/DBOW.PNG" width="400"/>
</p>

**Universal Sentence Encoder**

The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. The universal-sentence-encoder model is trained with a **Deep Averaging Network (DAN)** encoder.

<p align="center">
<img src="data/readme_pics/DAN.PNG" width="400"/>
</p>


### Clustering Techniques

In this project, I have used three different clustering techniques: **K-MEANS**, **DB-SCAN** and **EM**. 

**K-Means Clustering:**
K-Means is a popular unsupervised machine learning algorithm used for clustering data. It partitions a dataset into K distinct, non-overlapping clusters based on the similarity of data points. The algorithm works by iteratively assigning data points to the nearest cluster centroid and updating the centroids to minimize the within-cluster variance. K-Means is efficient and easy to implement but requires specifying the number of clusters (K) in advance, which can be a limitation.

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
DBSCAN is a density-based clustering algorithm that doesn't require specifying the number of clusters in advance. It identifies clusters as regions of high data point density separated by areas of lower density. DBSCAN groups data points that are close to each other into clusters and can also identify noise points that don't belong to any cluster. It's particularly useful for discovering clusters of arbitrary shapes and handling noisy data.

**EM (Expectation-Maximization) Clustering:**
EM is a statistical approach to clustering that models the data distribution as a mixture of several probability distributions. In EM clustering, it is assumed that data is generated from a mixture of Gaussian distributions. The algorithm iteratively estimates the parameters of these Gaussian distributions and assigns data points to the most likely component based on the estimated probabilities. EM clustering is capable of modeling clusters with different shapes and sizes and can handle situations where the clusters overlap.

**Comparison:**
Comparing the three techniques on the basis of cluster shape, size, noise, scalability and senstivity:

1) Cluster Shape: K-Means assumes clusters are spherical and equally sized. Struggles with non-spherical clusters. DBSCAN can identify clusters of arbitrary shapes. EM is capable of modeling clusters with different shapes.

2) Number of Clusters: K-Means requires specifying the number of clusters (K) in advance. DBSCAN automatically determines the number of clusters based on data density. EM requires specifying the number of components (clusters) in advance.

3) Handling Noise: K-Means is sensitive to outliers and can assign noise points to the nearest cluster. DBSCAN identifies noise points as outliers not belonging to any cluster. EM can handle noise to some extent but may assign noise points to a cluster if their likelihood is above a certain threshold.

4) Scalability: K-Means scales well to large datasets. DBSCAN can be computationally intensive for large datasets due to its density-based nature. EM is sensitive to the number of components and may not scale well for very large datasets.

5) Initialization Sensitivity: K-Meansis sensitive to the initial choice of centroids, may converge to suboptimal solutions. DBSCAN is not sensitive to initialization. EM is sensitive to initialization, can converge to different solutions based on the initial parameters.

### Custom Learning Rate Schedule

Similarly can you parse this one in `O(1)`?

<p align="left">
<img src="data/readme_pics/lr_formula.PNG"/>
</p>

Noup? So I thought, here it is visualized:

<p align="center">
<img src="data/readme_pics/custom_learning_rate_schedule.PNG"/>
</p>

It's super easy to understand now. Now whether this part was crucial for the success of transformer? I doubt it.
But it's cool and makes things more complicated. :nerd_face: (`.set_sarcasm(True)`)

*Note: model dimension is basically the size of the embedding vector, baseline transformer used 512, the big one 1024*

### Label Smoothing

First time you hear of label smoothing it sounds tough but it's not. You usually set your target vocabulary distribution
to a `one-hot`. Meaning 1 position out of 30k (or whatever your vocab size is) is set to 1. probability and everything else to 0.

<p align="center">
<img src="data/readme_pics/label_smoothing.PNG" width="700"/>
</p>

In label smoothing instead of placing 1. on that particular position you place say 0.9 and you evenly distribute the rest of
the "probability mass" over the other positions 
(that's visualized as a different shade of purple on the image above in a fictional vocab of size 4 - hence 4 columns)

*Note: Pad token's distribution is set to all zeros as we don't want our model to predict those!*

Aside from this repo (well duh) I would highly recommend you go ahead and read [this amazing blog](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar!

## Machine translation

Transformer was originally trained for the NMT (neural machine translation) task on the [WMT-14 dataset](https://torchtext.readthedocs.io/en/latest/datasets.html#wmt14) for:
* English to German translation task (achieved 28.4 [BLEU score](https://en.wikipedia.org/wiki/BLEU))
* English to French translation task (achieved 41.8 BLEU score)
 
What I did (for now) is I trained my models on the [IWSLT dataset](https://torchtext.readthedocs.io/en/latest/datasets.html#iwslt), which is much smaller, for the
English-German language pair, as I speak those languages so it's easier to debug and play around.

I'll also train my models on WMT-14 soon, take a look at the [todos](#todos) section.

---

Anyways! Let's see what this repo can practically do for you! Well it can translate!

Some short translations from my German to English IWSLT model: <br/><br/>
Input: `Ich bin ein guter Mensch, denke ich.` ("gold": I am a good person I think) <br/>
Output: `['<s>', 'I', 'think', 'I', "'m", 'a', 'good', 'person', '.', '</s>']` <br/>
or in human-readable format: `I think I'm a good person.`

Which is actually pretty good! Maybe even better IMO than Google Translate's "gold" translation.

---

There are of course failure cases like this: <br/><br/>
Input: `Hey Alter, wie geht es dir?` (How is it going dude?) <br/>
Output: `['<s>', 'Hey', ',', 'age', 'how', 'are', 'you', '?', '</s>']` <br/>
or in human-readable format: `Hey, age, how are you?` <br/>

Which is actually also not completely bad! Because:
* First of all the model was trained on IWSLT (TED like conversations)
* "Alter" is a colloquial expression for old buddy/dude/mate but it's literal meaning is indeed age.

Similarly for the English to German model.

## Setup

So we talked about what transformers are, and what they can do for you (among other things). <br/>
Let's get this thing running! Follow the next steps:

1. `git clone https://github.com/gordicaleksa/pytorch-original-transformer`
2. Open Anaconda console and navigate into project directory `cd path_to_repo`
3. Run `conda env create` from project directory (this will create a brand new conda environment).
4. Run `activate pytorch-transformer` (for running scripts from your console or set the interpreter in your IDE)

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies. <br/>
It may take a while as I'm automatically downloading SpaCy's statistical models for English and German.

-----

PyTorch pip package will come bundled with some version of CUDA/cuDNN with it,
but it is highly recommended that you install a system-wide CUDA beforehand, mostly because of the GPU drivers. 
I also recommend using Miniconda installer as a way to get conda on your system.
Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md)
and use the most up-to-date versions of Miniconda and CUDA/cuDNN for your system.

## Usage

#### Option 1: Jupyter Notebook

Just run `jupyter notebook` from you Anaconda console and it will open the session in your default browser. <br/>
Open `The Annotated Transformer ++.ipynb` and you're ready to play! <br/>

---

**Note:** if you get `DLL load failed while importing win32api: The specified module could not be found` <br/>
Just do `pip uninstall pywin32` and then either `pip install pywin32` or `conda install pywin32` [should fix it](https://github.com/jupyter/notebook/issues/4980)!

#### Option 2: Use your IDE of choice

You just need to link the Python environment you created in the [setup](#setup) section.

### Training

To run the training start the `training_script.py`, there is a couple of settings you will want to specify:
* `--batch_size` - this is important to set to a maximum value that won't give you CUDA out of memory
* `--dataset_name` - Pick between `IWSLT` and `WMT14` (WMT14 is not advisable [until I add](#todos) multi-GPU support)
* `--language_direction` - Pick between `E2G` and `G2E`

So an example run (from the console) would look like this: <br/>
`python training_script.py --batch_size 1500 --dataset_name IWSLT --language_direction G2E`

The code is well commented so you can (hopefully) understand how the training itself works. <br/>

The script will:
* Dump checkpoint *.pth models into `models/checkpoints/`
* Dump the final *.pth model into `models/binaries/`
* Download IWSLT/WMT-14 (the first time you run it and place it under `data/`)
* Dump [tensorboard data](#evaluating-nmt-models) into `runs/`, just run `tensorboard --logdir=runs` from your Anaconda
* Periodically write some training metadata to the console

*Note: data loading is slow in torch text, and so I've implemented a custom wrapper which adds the caching mechanisms
and makes things ~30x faster! (it'll be slow the first time you run stuff)*

### Inference (Translating)

The second part is all about playing with the models and seeing how they translate! <br/>
To get some translations start the `translation_script.py`, there is a couple of settings you'll want to set:
* `--source_sentence` - depending on the model you specify this should either be English/German sentence
* `--model_name` - one of the pretrained model names: `iwslt_e2g`, `iwslt_g2e` or your model(*)
* `--dataset_name` - keep this in sync with the model, `IWSLT` if the model was trained on IWSLT
* `--language_direction` - keep in sync, `E2G` if the model was trained to translate from English to German

(*) Note: after you train your model it'll get dumped into `models/binaries` see what it's name is and specify it via
the `--model_name` parameter if you want to play with it for translation purpose. If you specify some of the pretrained
models they'll **automatically get downloaded** the first time you run the translation script.

I'll link IWSLT pretrained model links here as well: [English to German](https://www.dropbox.com/s/a6pfo6t9m2dh1jq/iwslt_e2g.pth?dl=1) and [German to English.](https://www.dropbox.com/s/dgcd4xhwig7ygqd/iwslt_g2e.pth?dl=1)

That's it you can also visualize the attention check out [this section.](#visualizing-attention) for more info.

### Evaluating NMT models

I tracked 3 curves while training:
* training loss (KL divergence, batchmean)
* validation loss (KL divergence, batchmean)
* BLEU-4 

[BLEU is an n-gram based metric](https://www.aclweb.org/anthology/P02-1040.pdf) for quantitatively evaluating the quality of machine translation models. <br/>
I used the BLEU-4 metric provided by the awesome **nltk** Python module.

Current results, models were trained for 20 epochs (DE stands for Deutch i.e. German in German :nerd_face:):

| Model | BLEU score | Dataset |
| --- | --- | --- |
| [Baseline transformer (EN-DE)](https://www.dropbox.com/s/a6pfo6t9m2dh1jq/iwslt_e2g.pth?dl=1) | **27.8** | IWSLT val |
| [Baseline transformer (DE-EN)](https://www.dropbox.com/s/dgcd4xhwig7ygqd/iwslt_g2e.pth?dl=1) | **33.2** | IWSLT val |
| Baseline transformer (EN-DE) | x | WMT-14 val |
| Baseline transformer (DE-EN) | x | WMT-14 val |

I got these using greedy decoding so it's a pessimistic estimate, I'll add beam decoding [soon.](#todos)

**Important note:** Initialization matters a lot for the transformer! I initially thought that other implementations
using Xavier initialization is again one of those arbitrary heuristics and that PyTorch default init will do - I was wrong:

<p align="center">
<img src="data/readme_pics/bleu_score_xavier_vs_default_pt_init.PNG" width="450"/>
</p>

You can see here 3 runs, the 2 lower ones used PyTorch default initialization (one used `mean` for KL divergence
loss and the better one used `batchmean`), whereas the upper one used **Xavier uniform** initialization!
 
---

Idea: you could potentially also periodically dump translations for a reference batch of source sentences. <br/>
That would give you some qualitative insight into how the transformer is doing, although I didn't do that. <br/>
A similar thing is done when you have hard time quantitatively evaluating your model like in [GANs](https://github.com/gordicaleksa/pytorch-gans) and [NST](https://github.com/gordicaleksa/pytorch-nst-feedforward) fields.

### Tracking using Tensorboard

The above plot is a snippet from my Azure ML run but when I run stuff locally I use Tensorboard.

Just run `tensorboard --logdir=runs` from your Anaconda console and you can track your metrics during the training.

### Visualizing attention

You can use the `translation_script.py` and set the `--visualize_attention` to True to additionally understand what your
model was "paying attention to" in the source and target sentences.

Here are the attentions I get for the input sentence `Ich bin ein guter Mensch, denke ich.`

These belong to layer 6 of the encoder. You can see all of the 8 multi-head attention heads.

<p align="center">
<img src="data/readme_pics/attention_enc_self.PNG" width="850"/>
</p>

And this one belongs to decoder layer 6 of the self-attention decoder MHA (multi-head attention) module. <br/>
You can notice an interesting **triangular pattern** which comes from the fact that target tokens can't look ahead!

<p align="center">
<img src="data/readme_pics/attention_dec_self.PNG" width="850"/>
</p>

The 3rd type of MHA module is the source attending one and it looks similar to the plot you saw for the encoder. <br/>
Feel free to play with it at your own pace!

*Note: there are obviously some bias problems with this model but I won't get into that analysis here*



### Todos:

Finally there are a couple more todos which I'll hopefully add really soon:
* Explore how open source LLMs can be used to detect these scams.
* Make a data pipeline with Kedro and MLflow.
* Deploy the models and learn the effect of drift on their performance.





## Acknowledgements

I found these resources useful (while developing this one):

* [Detecting Telephone-based Social Engineering Attacks using Scam Signatures](https://dl.acm.org/doi/pdf/10.1145/3445970.3451152)
* [Universal Sentence Encoder](https://huggingface.co/Dimitre/universal-sentence-encoder)
* [Doc2Vec](https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)


## Citation

If you find this code useful, please cite the following:

```
@misc{Zain2023DetectingSocialEngineeringAttacks,
  author = {Zain, Abideen},
  title = {detect-social-engineering-attacks},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abideenml/ResearchProjects/tree/master/Detecting-Telephone-based-Social-Engineering-Attacks}},
}
```

## Connect with me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:

* Connect and reach me on [LinkedIn](https://www.linkedin.com/in/zaiinulabideen/) and [Twitter](https://twitter.com/zaynismm)
* Follow me on 📚 [Medium](https://medium.com/@zaiinn440)
* Subscribe to my 📢 weekly [AI newsletter](https://rethinkai.substack.com/)!

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/abideenml/ResearchProjects/blob/master/LICENCE)