1. Clasificator Automat De Categorii Pentru Produse 

Acest proiect dezvolta un model de invatare automata capabil sa prezica automat categoria de produs pe baza titlului sau. Solutia este utila pentru platformele de comert online care introduc zilnic mii de produse si au nevoie de clasificare rapida si precisa.

2. Structura Proiectului

Product_Category_Classifier/
    ─ data/
        products.csv
    ─ notebooks/
        01_eda_and_modeling.ipynb
    ─ models/
        product_classifier.pkl
    ─ src/
        train_model.py
        predict_category.py

    README.md
    requirements.txt

3. Obiectivul Proiectului

Scopul proiectului este de a automatiza clasificarea produselor in categorii precum:
- Mobile Phones
- Washing Machines
- Fridge Freezers
- Digital Cameras
- etc.
Modelul foloseste titlul produsului (ex: “iPhone 7 32GB Gold”) pentru a prezice categoria corecta.

4. Setul De Date

Fisierul products.csv contine peste 30.000 de produse reale, cu urmatoarele coloane:
- Product ID
- Product Title
- Merchant ID
- Category Label (eticheta tinta)
- Product Code
- Number of Views
- Merchant Rating
- Listing Date
Modelul foloseste in principal Product Title, dar pot fi adaugate si alte caracteristici.

5. Notebook-ul De Analiza
Notebook-ul "01_eda_and_modeling.ipynb" include:
- incarcarea si explorarea datelor
- curatarea datelor
- inginerie de caracteristici
- testarea mai multor modele ML
- evaluarea performantei
- salvarea modelului final in format .pkl

6. Antrenarea Modelului
Pentru a antrena modelul, ruleaza:
"python src/train_model.py"
Acest script:
- incarca datele
- curata titlurile si categoriile
- construieste un pipeline TF‑IDF + Logistic Regression
- antreneaza modelul
- evalueaza performanta
- salveaza modelul in "models/product_classifier.pkl"

7. Predictia Categoriei (mod interactiv)
Ruleaza:
"python src/predict_category.py"

Introdu un titlu de produs, de exemplu:
Titlul produsului: iphone 7 32gb gold
Categoria prezisa: Mobile Phones
Tasteaza exit pentru a inchide programul.

8. Evaluarea Modelului
Modelul este evaluat folosind:
- Acuratete
- Precision, Recall, F1-score
- Matrice de confuzie (in notebook)

9. Instalarea Dependentelor
Instaleaza librariile necesare:
pip install -r requirements.txt

Continutul fisierului requirements.txt:
pandas
numpy
scikit-learn
matplotlib
seaborn

10. Rezultatele Modelului
Modelul final a fost antrenat folosind un pipeline TF‑IDF + Logistic Regression. Performanta a fost evaluata pe un set de testare reprezentand 20% din date.
Raportul de clasificare include:
- Precision, Recall si F1-score pentru fiecare categorie
- media macro si weighted
- analiza dezechilibrelor dintre clase
Matricea de confuzie a fost generata in notebook si arata distributia predictiilor corecte si gresite pentru fiecare categorie.


11. Imbunatatiri Posibile
- folosirea unui model LinearSVC pentru performanta mai buna
- tuning de hiperparametri
- adaugarea de noi features (lungimea titlului, detectarea brandului etc.)
- implementarea unei API (FastAPI / Flask)
- interfata grafica pentru utilizator

Proiect realizat de Mutascu Georgiana.
Acest proiect este destinat uzului educational.
