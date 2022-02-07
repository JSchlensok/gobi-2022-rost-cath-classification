# GoBi 2022 CATH Classification

## Resources
Rolling slide deck: https://docs.google.com/presentation/d/1TvTKibsBg_XdaENP_taODn4jPAQi8Y8t9tuwYYHxTKs/edit?usp=sharing
https://www.cathdb.info/

### Papers
ProtT5 papers:
https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8

Method review paper: https://academic.oup.com/bib/article/19/2/231/2562645?login=true

CATH papers:


### Code Examples
https://github.com/sacdallago/bio_embeddings/blob/develop/notebooks/deeploc_machine_learning.ipynb
https://githubplus.com/Rostlab/EAT

## Data
Download the following data:

- Sequences Training: https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/train74k.fasta
- Sequences Validation: https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/val200.fasta
- Sequences Test: https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/test219.fasta

- Labels: https://rostlab.org/~deepppi/cath-domain-list.txt

- Embeddings: https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.h5


## Requirements
1. Install requirements with:
```
pip install -r requirements.txt
```
2. When adding new imports that aren't saved in requirements.txt yet, add them to file with:
```
pip freeze > requirements.txt
```


## Code Formatting
Before committing your code run the following command:
```
black src
```
Configurations for formatting with black are saved in [black configuration file](src/pyproject.toml).
More information in [Black documentation](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#configuration-via-a-file).


## Training in Google Colab
1. Create a personal access token as described in [GitHub Docs Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
2. Open Google Colab Notebook and clone the repository with:
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
```
3. Install requirements:
```
!pip install -r /content/gobi-2022-rost-cath-classification/requirements.txt
```
4. Download data to data directory:
```
!wget -P /content/gobi-2022-rost-cath-classification/data https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/train74k.fasta
!wget -P /content/gobi-2022-rost-cath-classification/data https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/val200.fasta
!wget -P /content/gobi-2022-rost-cath-classification/data https://github.com/Rostlab/EAT/blob/main/data/ProtTucker/test219.fasta
!wget -P /content/gobi-2022-rost-cath-classification/data https://rostlab.org/~deepppi/cath-domain-list.txt
!wget -P /content/gobi-2022-rost-cath-classification/data https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.h5
```
5. Run your training script:
```
!python /content/gobi-2022-rost-cath-classification/src/gobi_cath_classification/scripts_charlotte/train_eval.py
```



