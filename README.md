# GoBi 2022 CATH Classification
A comparison of different machine learning approaches for predicting CATH superfamily labels of protein sequences from their ProtT5 embeddings. This was done in the scope of the project part of the [Rostlab](https://rostlab.org) GoBi 2021/2022.

## Resources
- Slides:
    - [Intro presentation](https://docs.google.com/presentation/d/1TvTKibsBg_XdaENP_taODn4jPAQi8Y8t9tuwYYHxTKs/edit?usp=sharing)
    - [Intermediate presentation](https://docs.google.com/presentation/d/1_3Y6vuqYIWZX2Ip2UnL3sXAbx9C4tHZRkw7Z2e44vWg/edit?usp=sharing)
    - [Final presentation](https://docs.google.com/presentation/d/1z13lF1WeNIKjIAZgW2vlUoawp6SchPlB3WkiE5Q01-4/edit?usp=sharing)

- **[Final report](https://www.overleaf.com/2953961862zbsyvgtsngst)** (this gives a great overview!)
- [Trello board](https://trello.com/b/iEvimTbs/gobi-praktikum)

### Papers
- [ProtT5](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3)
- [SeqVec](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
- [Remote homology detection method review](https://academic.oup.com/bib/article/19/2/231/2562645?login=true)

#### CATH info:
- https://www.cathdb.info/

### Code Examples
- https://github.com/sacdallago/bio_embeddings/blob/develop/notebooks/deeploc_machine_learning.ipynb

### Reference method: ProtTucker
- [paper](https://www.biorxiv.org/content/10.1101/2021.11.14.468528v1)
- [repo](https://githubplus.com/Rostlab/EAT)

## Data
- Sequences
    - [Training](https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/train66k.fasta)
    - [Validation](https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/val200.fasta)
    - [Test](https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/test219.fasta)
    - [Lookup](https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/lookup69k.fasta)
- [Labels](https://rostlab.org/~deepppi/cath-domain-list.txt)
- [Embeddings](https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.h5)


## Requirements
Install requirements using:
```
pip install -r requirements.txt
```


## Code Formatting
Before committing your code run the following command:
```
black src
```
Configurations for formatting with black are saved in the [Black configuration file](pyproject.toml).

More information in the [Black documentation](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#configuration-via-a-file).


## Training in Google Colab
1. Create a personal access token as described in [GitHub Docs Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

2. Open Google Colab Notebook and copy the following snippet and execute it:
```python
from getpass import getpass
import urllib
import subprocess

def clone_dialog():
    token = getpass('GitHub personal access token: ')
    token = urllib.parse.quote(token) # converted into url format
    
    output = subprocess.run(
        ["git", "clone", f'https://{token}@github.com/JSchlensok/gobi-2022-rost-cath-classification.git'], 
        capture_output=True, text=True
    ).stderr
    print(f"output: {output}")
    

clone_dialog()

# Install requirements
!pip install -r /content/gobi-2022-rost-cath-classification/requirements.txt

# Add your module to PYTHONPATH
!echo $PYTHONPATH
%env PYTHONPATH=$PYTHONPATH:/content/gobi-2022-rost-cath-classification/src
!echo $PYTHONPATH

# Download data
!wget -P /content/gobi-2022-rost-cath-classification/data https://raw.githubusercontent.com/Rostlab/EAT/main/data/ProtTucker/train66k.fasta
!mv /content/gobi-2022-rost-cath-classification/data/train66k.fasta /content/gobi-2022-rost-cath-classification/data/train74k.fasta
!wget -P /content/gobi-2022-rost-cath-classification/data https://raw.githubusercontent.com/Rostlab/EAT/main/data/ProtTucker/test219.fasta
!wget -P /content/gobi-2022-rost-cath-classification/data https://raw.githubusercontent.com/Rostlab/EAT/main/data/ProtTucker/val200.fasta
!wget -P /content/gobi-2022-rost-cath-classification/data https://rostlab.org/~deepppi/cath-domain-list.txt
!wget -P /content/gobi-2022-rost-cath-classification/data https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.h5
```
3. Run a script of your choice, e.g.:
```
!python /content/gobi-2022-rost-cath-classification/src/gobi_cath_classification/pipeline/train_eval.py
```

## Reproduce results from final report
1. First, open a new Notebook in Google Colab and follow the instructions 1. and 2. in the "Training in Google Colab" Section.
2. Execute code snippet for a model of your choice, listed below:

### Logistic Regression:
The 1st model in the following trial (Trial name = training_function_xxxxx_00000) is the final Log Reg model.
```
%cd /content/gobi-2022-rost-cath-classification/
!git checkout 207edeb7 
!python /content/gobi-2022-rost-cath-classification/src/gobi_cath_classification/scripts_charlotte/train_nn.py
```

### FCNN:
The 3rd model in the following trial (Trial name = training_function_xxxxx_00002) is the final FCNN model.
```
%cd /content/gobi-2022-rost-cath-classification/
!git checkout e215ef7d 
!python /content/gobi-2022-rost-cath-classification/src/gobi_cath_classification/scripts_charlotte/train_nn.py
```

### Distance Model:
The first and only run in this trial is the final model.
```
%cd /content/gobi-2022-rost-cath-classification/
!git checkout d09d26cb
!python /content/gobi-2022-rost-cath-classification/src/gobi_cath_classification/scripts_charlotte/train_nn.py
```

### Gaussian Naive Bayes and Random Forest:
1. run: Gaussian Naive Bayes
2. run: Random Forest
```
%cd /content/gobi-2022-rost-cath-classification/
!git checkout f2b958c5
!python /content/gobi-2022-rost-cath-classification/src/gobi_cath_classification/pipeline/train_eval.py
```

### ArcFace
Follow the steps in the `ArcFace Runner` notebook to either train your own model or load the best one (`arcface_2022-04-06_1403_val_acc_0.65`, loaded by default)

