Navrhnuté siete sa nachádzajú v priečinku src/networks.py

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



### BaseConv 
Táto sieť používa 4 obyčajné konvolučné vrstvy. Každá konvolučná vrstva je nasledovaná  MaxPooling vrsrvou s kernelom (2,2). Za posledným MaxPoolingom sa nachádza Droupot vrstva(0.25).

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
