# Dokumentácia riešenia
Tento dokument obsahuje popis riešenia nášho projektu.

## Dáta
Naše dáta pochádzajú zo Stanford Dog Datasetu, sú to obrázky získané z ImageNet databázy. Dataset obsahuje 22126 obrázkov psov 120 rôznych plemien.

## Preprocessing
Každý obrázok obsahoval aspoň jedného psa, pričom pre každého psa na obrázku boli uvedené bounding box súradnice, podľa ktorých sme psov z obrázkov vyrezávali. Následne sme každý vyrezaný obrázok interpolovali na veľkost 299x299 pixelov.
![Bouding boxes](bounding_boxes.png)

## Použité architektúry
Celkovo sme použili 4 architektúry z čoho 2 boli naše vlastné. Prvou našou sieťou bola obyčajná konvolučná sieť so 4 konvolučnými vrstvami. Každá konvolučná vrstva je nasledovaná MaxPooling vrsrvou s kernelom (2,2). Za posledným MaxPoolingom sa nachádza Droupot vrstva(0.25).

# BaseConv 
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
Táto sieť je našou verziou InceptionNetu. Používa 4 inception vrstvy. Každú Inception vrstvu nasleduje MaxPooling vrsrva s kernelom (2,2).
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




