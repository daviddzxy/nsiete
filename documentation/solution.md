# Dokumentácia riešenia
Tento dokument obsahuje popis riešenia nášho projektu. Návod na spustenie skriptov sa nachádza v README.md v roote projektu.

## __Dáta__
Naše dáta pochádzajú zo Stanford Dog Datasetu, sú to obrázky získané z ImageNet databázy. Dataset obsahuje 22126 obrázkov psov 120 rôznych plemien.

## __Predspracovanie__
Každý obrázok obsahoval aspoň jedného psa, pričom pre každého psa na obrázku boli uvedené bounding box súradnice, podľa ktorých sme psov z obrázkov vyrezávali. Následne sme každý vyrezaný obrázok interpolovali na veľkost 299x299 pixelov.
![Bouding boxes](bounding_boxes.png)

## __Použité architektúry__
Celkovo sme použili štyri architektúry z čoho dve boli naše vlastné. 

### BaseConv 
Prvou našou sieťou bola obyčajná konvolučná sieť so štyrmi konvolučnými vrstvami. Každá konvolučná vrstva je nasledovaná MaxPooling vrsrvou s kernelom (2,2). Za posledným MaxPoolingom sa nachádza Droupot vrstva(0.25).

```
Layer (type)                            
================================
conv2d (Conv2D)                   
________________________________
max_pooling2d (MaxPooling2,2)     
________________________________
conv2d_1 (Conv2D)            
________________________________
max_pooling2d_1 (MaxPooling2,2)   
________________________________
conv2d_2 (Conv2D)            
________________________________
max_pooling2d_2 (MaxPooling2,2)   
________________________________
conv2d_3 (Conv2D)            
________________________________
max_pooling2d_3 (MaxPooling2,2)     
________________________________
dropout (Dropout 0,25)            
________________________________
flatten (Flatten)            
________________________________
dense (Dense)                 
________________________________
dense_1 (Dense)                
================================
```
### Inception
Druhou použitou sieťou je naša verzia Inception siete, ktorá využívala Inception vrstvy. Celkovo sa sieť skladá zo štytorch takýchto Inception vrstviev. Každú Inception vrstvu nasleduje MaxPooling vrsrva s kernelom (2,2).

![Inception layer](inception_layer.png)

```
Layer (type)                 
================================
inception_layer (Inception)  
________________________________
max_pooling2d_1 (MaxPooling2,2)      
________________________________
inception_layer_1 (Inception)    
________________________________
max_pooling2d_3 (MaxPooling2,2)    
________________________________
inception_layer_2 (Inception   
________________________________
max_pooling2d_5 (MaxPooling2,2)      
________________________________
inception_layer_3 (Inception)    
________________________________
max_pooling2d_7 (MaxPooling2,2) 
________________________________
flatten (Flatten)            
________________________________
dense (Dense)                  
________________________________
dense_1 (Dense)                   
================================
```

### InceptionV3
[Táto sieť je preddefinovaná v knižnici Keras](https://keras.io/applications/#inceptionv3). Používali sme jej nepredtrénovanú verziu. Táto verzia siete obsahuje taktiež Inception vrstvy ako naša navrhovaná sieť, avšak nepoužíva 5x5 konvolúciu, ale dve 3x3 konvolúcie, čím sa zmenšil počet trénovateľných parametrov siete. Celkovo má táto sieť 42 vrstiev. Bližšie detajly k tejto sieti je možné získať [tu](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf).

![InceptionV3](inceptionV3.png)

### InceptionResNetV2
InceptionResNetV2 obsahuje Inception vrstvy, ktoré sú oproti InceptionV3 zjednodušené, taktiež je táto sieť oproti V3 hlbšia. Hlavným vylepšením tejto siete sú reziduálne spojenia,ktoré napomáhajú pri trénovaní extrémne hlbokých sietí. Celá špecifikácie siete je [tu](https://arxiv.org/pdf/1602.07261.pdf).

![InceptionV3](inception_res_net.png)

## __Trénovanie__
Na trénovanie sme používali 80% obrázkov z celej množiny, validovali sme na ostatných 20% obrázkov. Pri trénovaní sme taktiež využili augmentáciu s nasledujúcimi parametrami:

```
horizontal_flip=True,
rotation_range=30,
zoom_range=0.15,
shear_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
fill_mode="nearest"
```

Príklad augmentovaných obrázkov(prvý obrázok je originál):
![Augmentácia](augmentation.png)

Na zastavenie trénovania sme používali techniku skorého zastavenia, ktoré sledovalo hodnotu validačnej loss funkcie, ak sa hodnota loss funkcie nezlepšila 5 epóch za sebou, tak sa trénovanie zastavilo a váhy modelu sa nastavili na posledné s najlepšou hodnotou validačnej loss funkcie.

## __Experimenty__
### InceptionV3 experiment
```
batch_size=64,
learning_rate=0.0005,
augmentation=True
```
Experiment, pri ktorom sme použili model InceptionV3 sme trénovali celkovo viac ako osem hodín na karte Nvidia Tesla K80. Model sa trénoval celkovo 31 epoch, pričom najlepšia dosiahnutá presnosť na validačných dátach bola 0.63%. Logy z trénovania sa nachádzajú v priečinku logs/InceptionV3

## __Záver__
Natrénovali sme celkovo 4 modely, z ktorých najlepší bol InceptionResnetV2. V pláne sme mali vyskúšať aj transfer learning, ale Keras obsahuje len váhy, ktoré boli natrénované na ImageNet databáze, z ktorej máme dáta aj my a teda niektoré triedy obrázkov sú rovnaké. Taktiež sme mali v pláne prehľadávať hyper parametre, avšak kvôli vysokej časovej náročnosti trénovania modelov, sme tento krok vynechali.
