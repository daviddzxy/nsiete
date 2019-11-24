Navrhnuté siete sa nachádzajú v priečinku src/networks.py

### Inception
Táto sieť je našou verziou InceptionNetu. Používa 4 inception vrstvy. Každú Inception vrstvu nasleduje MaxPooling vrsrva s kernelom (2,2).

### BaseConv 
Táto sieť používa 4 obyčajné konvolučné vrstvy. Každá konvolučná vrstva je nasledovaná  MaxPooling vrsrvou s kernelom (2,2). Za posledným MaxPoolingom sa nachádza Droupot vrstva(0.25).
