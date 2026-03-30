import pickle

def load_model(path="models/product_classifier.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def main():
    # Incarcarea modelului
    model = load_model()
    print("Model incarcat cu succes.")
    print("Introdu un titlu de produs pentru a prezice categoria.")
    print("Tasteaza 'exit' pentru a iesi.\n")

    # Bucla interactiva pentru predictii
    while True:
        title = input("Titlul produsului: ")

        if title.lower() in ["exit", "quit"]:
            print("Iesire din program...")
            break

        prediction = model.predict([title])[0]
        print(f"Categoria prezisa: {prediction}\n")

if __name__ == "__main__":
    main()