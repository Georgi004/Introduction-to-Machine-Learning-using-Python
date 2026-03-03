import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Incarcarea datelor
# citim fisierul CSV care contine mesajele si etichetele (spam/ham).
# este important ca fisierul sa fie un acelasi folder cu scriptul
# programul va da eroare daca numele fisierului este gresit
df = pd.read_csv("message.csv")

print("Primele randuri din setul de date:")
print(df.head()) # verificam daca datele sunt incarcate corect.

# Curatarea datelor
# eliminam randurile care nu au categorie sau mesaj, deoarece modelul nu poate invata din date incomplete.
df = df.dropna(subset=["category", "message"])
# transformam categoriile in litere mici pentru a evita diferentele intre 'Spam', 'spam', 'SPAM' etc.
df["category"] = df["category"].str.lower()
# pastram doar valorile valide: 'spam', 'ham'.
# uneori seturile de date pot contine etichete gresite.
df = df[df["category"].isin(["spam", "ham"])]

# Separarea in X si Y
# x - mesajele(datele de intrare), y - etichetele(spam sau ham)
# aceasta separare este standard in machine learning.
X = df["message"]
Y = df["category"]

# Vectorizarea textului
# folosim TfidfVectorizer (transforma textul in vectori numerici), deoarece modelele de ML nu pot lucra cu text brut.
# aici modelul invata ce cuvinte sunt importante pentru spam.
# ATENTIE: modelul invata doar din cuvintele din setul de date, daca utilizatorul introduce mesaje in alta limba(ex:romana), modelul poate clasifica gresit. 
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Antrenarea modelului
# LogisticRegression este un model rapid si eficient pentru clasificare binara.
# max_iter = 1000 previne erorile de convergenta.
model = LogisticRegression(max_iter=1000)
model.fit(X_vectors, Y)

print("\nModelul a fost antrenat cu succes!")

# Interfata interactiva
# aceasta parte permite utilizatorului sa testeze modelul cu mesaje noi.
# folosim o bucla infinita care se opreste doar cand utilizatorul scrie 'exit'.
print("\nIntrodu un mesaj pentru a verifica daca este spam.")
print("Tasteaza 'exit' pentru a inchide programul.\n")

while True:
    user_input = input("Mesaj: ")

    # calea de iesire din bucla

    if user_input.lower() == "exit":
        print("Program inchis. La revedere!")
        break

    # transformam mesajul utilizatorului in vector numeric.
    # trebuie sa folosim EXACT acelasi vectorizer ca la antrenare
    # daca am folosi un vectorizer nou aici, modelul nu ar intelege structura numerica si ar da eroare.
    user_vector = vectorizer.transform([user_input])
    # modelul prezice categoria mesajului
    prediction = model.predict(user_vector)[0]

    # afisam rezultatul in mod clar
    if prediction == "spam":
        print("Acest mesaj este clasificat ca: SPAM\n")
    else:
        print("Acest mesaj NU este spam\n")

# Observatii
# 1. cea mai grea parte a fost sa inteleg cum transforma TF-IDF textul in numere si de ce are nevoie modelul de aceasta etapa.
# 2. am testat mesaje in romana si am realizat ca modelul nu le recunoaste ca spam, de aici am dedus faptul ca el invata doar din limba in care este antrenat.
# 3. alta "problema" a fost partea interactiva, unde am realizat ca trebuie sa folosesc acelasi vectorizer pentru transformarea mesajelor noi.
# 4. pentru o imbunatatire a acestui model, as adauga in viitor mesaje spam si ham in limba romana, pentru ca utilizatorii reali sa poata testa mesajele in limba lor.
