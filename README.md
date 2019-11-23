# NSIETE projekt - Klasifikácia plemien psov

**Autori:** Dávid Baláž, Richard Šmajda **Čas cvičenia:** Utorok 11.00 hod
**Cvičiaci:** Ing. Matúš Pikuliak

### Spustenie projektu:

```
1. git clone https://github.com/daviddzxy/nsiete.git
2. cd nsiete
3. curl https://transfer.sh/188VD/raw.zip --output raw.zip && unzip -jnq raw.zip raw/* -d data/raw/ && rm raw.zip
4. docker-compose -f docker/docker-compose.yml up --no-start
5. ./run_docker_sh
6. python src/preprocessing.py
7. python src/train.py
```

### Štruktúra projektu:
```
nsiete/
├── data
│   ├── annotations.csv # csv s metainformáciami a anotáciami
│   ├── processed # predspracované obrázky
│   └── raw # neupravené obrázky
├── docker
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
├── logs
├── model_weights # priečinok na ukladanie váh modelov
├── notebooks
│   └── data_analysis.ipynb
├── proposal
│   ├── doggo.png
│   └── proposal.md
├── README.md
├── run_docker.sh # skript na spustenie docker containera 
└── src
    ├── networks.py # definície neurónových sietí a vrstiev
    ├── preprocessing.py # skript na spustenie preprocessingu
    └── train.py # skript na spustenie trénovania
```





