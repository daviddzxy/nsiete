NSIETE projekt - Klasifikácia plemien psov
==========================================

**Autori:** Dávid Baláž, Richard Šmajda **Čas cvičenia:** Utorok 11.00 hod
**Cvičiaci:** Ing. Matúš Pikuliak

### Spustenie projektu:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. git clone https://github.com/daviddzxy/nsiete.git
2. cd nsiete
3. curl https://transfer.sh/188VD/raw.zip --output raw.zip && unzip -jnq raw.zip raw/* -d data/raw/ && rm raw.zip
4. docker-compose -f docker/docker-compose.yml up --no-start
5. ./run_docker_sh
6. python src/preprocessing.py
7. python src/train.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Štruktúra projektu:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nsiete/
├── data
│   ├── annotations.csv # csv s metainformáciami a anotáciami
│   ├── processed # predspracované obrázky
│   └── raw # neupravené obrázky
├── docker
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
├── documentation
│   ├── doggo.png
│   ├── inception_layer.png
│   ├── proposal.md
│   └── solution.md
├── logs # priečinok na ukladanie záznamov trénovania
├── model_weights # priečinok na ukladanie váh modelov
├── notebooks
│   └── data_analysis.ipynb
├── README.md
├── run_docker.sh # skript na spustenie docker containera 
└── src
    ├── networks.py # definície neurónových sietí a vrstiev
    ├── preprocessing.py # skript na spustenie preprocessingu
    └── train.py # skript na spustenie trénovania
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Použitie:

train.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Training script. [-h] [-e EPOCHS] [-l LEARNING_RATE] [-b BATCH_SIZE]
                        [-n {Inception,BaseConv}] [-s SPLIT] [-w]
                        [-d [DOG_BREEDS [DOG_BREEDS ...]]]

optional arguments:
  -h, --help            Show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Sets number of epochs. (default: 10)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Sets learning rate. (default: 0.0001)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Sets batch size. (default: 32)
  -n {Inception,BaseConv}, --network {Inception,BaseConv}
                        Type of network. (default: Inception)
  -s SPLIT, --split SPLIT
                        Portion of dataset used for training. Default=0.8.
                        (default: 0.8)
  -w, --workaround      Turn on workaround for Error "Cudnn could not create
                        handle" because of low memory. Run only if you train
                        the model on low spec GPU. Workaround is turned off by
                        default, to turn it on set the -w argument (default:
                        False)
  -d [DOG_BREEDS [DOG_BREEDS ...]], --dog-breeds [DOG_BREEDS [DOG_BREEDS ...]]
                        List of dog breeds to train on the neural network. Use
                        the names from column names from annotaions.csv. If
                        not specified train on all breeds.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

preprocessing.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Preprocessing of images in /data/raw/. Cuts out dogs from images and resizes them. 
Preprocessed images are stored in /data/processed.
       [-h] [-r RESIZE]

optional arguments:
  -h, --help            Show this help message and exit
  -r RESIZE, --resize RESIZE
                        Width and height of resized image.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
