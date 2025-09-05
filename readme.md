# Analiza in vizualizacija podatkov o cvetnem prahu

Ta projekt je delovni potek v Pythonu, zasnovan za avtomatizirano analizo in vizualizacijo podatkov o koncentraciji cvetnega prahu. Program prebere surove podatke iz Excelovih datotek, izvede vrsto podrobnih analiz in ustvari različne grafične prikaze, ki ponazarjajo trende, popolnost podatkov in ključne meritve za določene regije.

Projekt je zasnovan za obravnavo podatkov iz več lokacij, kot so Ljubljana, Maribor in Primorje, vendar ga je mogoče razširiti na druge lokacije, če so podatki na voljo v ustreznem formatu.

## Funkcionalnosti
Obdelava podatkov: Uvozi surove podatke o koncentraciji cvetnega prahu iz Excelovih datotek (.xlsx) [cite: src/data_loader.py]. Podatke očisti, uredi manjkajoče vrednosti in standardizira datumske formate [cite: src/data_loader.py].

Analiza popolnosti podatkov: Ustvari vizualizacije, ki prikazujejo popolnost podatkov glede na leto, mesec in vrsto cvetnega prahu [cite: src/plotting.py].

Analiza sezone cvetnega prahu: Analizira ključne meritve, kot so K10, K50 in K90, ki določajo začetek, sredino in konec sezone cvetenja [cite: src/analysis.py]. Omogoča tudi določitev začetka (5%) in konca (95%) sezone glede na kumulativno vsoto v izbranem referenčnem letu [cite: src/analysis.py].

Korelacijska analiza med regijami: Izračuna in prikaže korelacijo med koncentracijami istih vrst cvetnega prahu v različnih regijah [cite: src/analysis.py].

Grafični prikazi: Ustvarja različne diagrame, vključno s toplotnimi kartami popolnosti podatkov in podrobnimi grafi za vsako vrsto cvetnega prahu [cite: src/plotting.py]. Posebej vključuje kombinirane grafe, ki združujejo časovne vrste in korelacijsko toplotno karto za boljšo vizualizacijo odnosov med regijami [cite: src/plotting.py].

Struktura datotek
cvetni_prah_v2/
├── data/
│   └── raw/              # Mesto za surove podatkovne datoteke (npr. 'Ljubljana2024.xlsx')
├── results/              # Mapa, kjer so shranjeni ustvarjeni grafi in analitične datoteke
├── src/
│   ├── analysis.py       # Izvaja statistične analize in izračune trendov.
│   ├── data_loader.py    # Odgovoren za nalaganje in čiščenje podatkov.
│   ├── plotting.py       # Vsebuje funkcije za ustvarjanje grafov.
│   └── utils.py          # Pomožne funkcije (npr. drseče povprečje, obdelava poti).
├── .gitignore            # Določa datoteke in mape, ki jih Git ignorira.
└── main.py               # Glavni skript, ki usmerja delovni potek analize.

Predpogoji
Za zagon projekta so potrebne naslednje knjižnice Python:

pandas [cite: src/data_loader.py]

numpy [cite: src/utils.py]

matplotlib [cite: src/utils.py]

seaborn [cite: src/plotting.py]

tqdm [cite: src/data_loader.py]

openpyxl (ali druga knjižnica za branje .xlsx datotek) [cite: src/data_loader.py]

Knjižnice lahko namestite z uporabo pip:

pip install pandas numpy matplotlib seaborn tqdm openpyxl

Navodila za uporabo
Priprava podatkov: Vse surove podatkovne datoteke (v formatu .xlsx) shranite v mapo data/raw/. Imena datotek morajo biti skladna z lokacijo (npr. Ljubljana2024.xlsx).

Zagon analize: V terminalu zaženite glavni skript:

python main.py

Skript bo samodejno obdelal podatke za vse navedene lokacije, ustvaril grafe in shranil rezultate v ustrezne podmape v mapi results/. Če podatkovna datoteka za določeno lokacijo ne obstaja, jo bo skript preskočil in nadaljeval [cite: main.py].