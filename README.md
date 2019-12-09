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
├── data
│   ├── annotations.csv # csv s metainformáciami a anotáciami, train.csv + valid.csv = annotations.csv
│   ├── processed # predspracované obrázky
│   ├── raw # neupravené obrázky
│   ├── train.csv 
│   └── valid.csv
├── docker
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
├── documentation
│   ├── proposal.md
│   └── solution.md # dokumentácia riešenia
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
                        [-n {Inception,InceptionV3,BaseConv,InceptionResNet}]
                        [-w] [-d [DOG_BREEDS [DOG_BREEDS ...]]] [-a]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Sets number of epochs. (default: 10)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Sets learning rate. (default: 0.0001)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Sets batch size. (default: 32)
  -n {Inception,InceptionV3,BaseConv,InceptionResNet}, --network {Inception,InceptionV3,BaseConv,InceptionResNet}
                        Type of network. (default: Inception)
  -w, --workaround      Turn on workaround for Error "Cudnn could not create
                        handle" because of low memory. Run only if you train
                        the model on low spec GPU. Workaround is turned off by
                        default, to turn it on set the -w argument (default:
                        False)
  -d [DOG_BREEDS [DOG_BREEDS ...]], --dog-breeds [DOG_BREEDS [DOG_BREEDS ...]]
                        List of dog breeds to train on the neural network. Use
                        the names from column names from annotaions.csv. If
                        not specified train on all breeds. (default: None)
  -a, --augmentation    Allow augmentation. (default: False)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

preprocessing.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Preprocessing of images in /data/raw/. Cuts out dogs from images and resizes them. 
Preprocessed images are stored in /data/processed.
       [-h] [-r RESIZE]

optional arguments:
  -h, --help            Show this help message and exit
  -r RESIZE, --resize RESIZE
                        Width and height of resized image. (default: 299)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
