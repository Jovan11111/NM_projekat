import matplotlib.pyplot as plt

# Broj elemenata svake klase
broj_elemenata = [9992, 10000, 9298, 1406, 9999]

# Imena klasa
klase = ['Slikarstvo', 'Fotografija', 'Šema', 'Skica', 'Tekst']

# Kreiranje grafika
plt.bar(klase, broj_elemenata, color='skyblue')

# Dodavanje imena na ose
plt.title('Broj elemenata po klasi')
plt.xlabel('Klase')
plt.ylabel('Broj elemenata')

# Prikazivanje grafika
plt.show()