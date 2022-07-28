import pandas as pd


def analyze_input():

    dataset = pd.read_json('./outputs/length_analyzer.json', typ='series')

    print(dataset)
    total = 0
    # Itero sobre el Dataset y lo fragmento
    for key, value in dataset.items():
        total += value

    print("Proporción promedio de tamaño de los resúmenes: " + str(total/len(dataset)))


def main():
    analyze_input()


if __name__ == "__main__":
    main()
