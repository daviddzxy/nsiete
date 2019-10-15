Návrh zadania
=============

Motivácia
---------
V našej práci budeme klasifikovať plemená psov. Klasifikácia objektov v obrázkoch pomocou konvolučných neurónových sietí je pomerne prebádaná oblasť. V našom projekte sa budeme snažiť overiť viacero postupov a architektúr strojového učenia použitého pri klasifikácii obrázkov. Pomocou moderných techník sa budeme pokúšať dosiahnúť výsledky porovnateľné s výsledkami súčasných metdód.

Súvisiaca práca
---------------

Obdobným problémom a to klasifikáciou plemien psov sa už zaoberali viaceré
práce. Vo väčšine týchto prác sa použil model konvolučnej neurónovej siete,
ktorý sa použil na klasifikáciu dát. Existujú práce zaoberajúce sa témou
klasifikácie plemien psov, ktoré pri tvorbe svojho modelu na klasifikáciu
využívali už existujúci model ResNet50 z knižnice Keras. Tento už natrénovaný
model ďalej trénovali. Pri trénovaní jedna z prác vykonala kvôli časovej
zložitosti výpočtov iba 20 epoch, pri ktorých už pomocou metriky správnosť
(angl. accuracy) rovnej 4.5% pozorovali učenie sa modelu. Pri vykonaní 250 epoch
predpokladali správnosť modelu na úrovni 40%.

Pri tejto práci bola použitá dátová množina, ktorá obsahovala 133 rôznych
plemien psov. Cieľom tejto práce bolo, rozoznanie či sa na obrázku jedná o
človeka, alebo psa, kde v prípade ak sa detekoval pes tak sa klasifikovala aj
jeho rasa.

Dataset
-------

Pri riešení nášho projektu budeme používať [Stanford dog
dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), ktorý obsahuje
20,580 obrázkov 120 plemien psov. Tento odataset obsahuje anotácie a obrázky z
ImageNet-u, čo je databáza obrázkov navrhnutá na výskum v oblasti rozpoznávania
obrázkov.

Samotné dáta obsahujú obrázky rôznej veľkosti vo formáte jpg a ďalšie doplňujúce
informácie v xml formáte ako plemeno psa, veľkosť obrázku a bounding box.

![Obrzok z datasetu](media/e014b59f37d30ce320e08649fe37fa00.png)

Obrzok z datasetu

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<annotation>
    <folder>02106166</folder>
    <filename>n02106166_1031</filename>
    <source>
        <database>ImageNet database</database>
    </source>
    <size>
        <width>500</width>
        <height>372</height>
        <depth>3</depth>
    </size>
    <segment>0</segment>
    <object>
        <name>Border_collie</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>11</xmin>
            <ymin>46</ymin>
            <xmax>376</xmax>
            <ymax>334</ymax>
        </bndbox>
    </object>
</annotation>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Návrh riešenia
-----------------
Náš prvý krok riešenia bude spočívať v analýze poskytnutých dát. Pokračovať budeme predspracovaním dát, čo bude zahŕňat prepísanie anotácii z XML formátu do csv formátu, orezanie obrázkov pomocou bounding box súradníc a normálizácii dát. Následne budeme pokračovať so samotným trénovaním siete. Pri trénovaní vyskúšame našu vlastnú architektúru siete a aj iné súčasné architektúry využívajúce sa pri klasifikácii obrázkov. Pri trénovaní použijeme techniku prehľadávania hyperparametrov gridsearch. Pri používaní iných architektúr môžeme skúsiť použiť aj techniku transfer learning a porovnať výsledky s našou architektúrou. Výsledky použitých architektúr následne porovnáme a vyvodíme z nich závery.
