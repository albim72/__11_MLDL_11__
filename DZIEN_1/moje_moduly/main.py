# import dane
# import dane as dn
from dane import nrfilii as nf
from dane import book as bk
from moje_funkcje.collectfunctions import czytaj_liste, czytaj_slownik

print(f'___________ kolekcje z modułu: dane __________')
print(nf)
print(bk)

print(f'___________ kolekcje z modułu: dane -> wyświetlone przez funkcje __________')
print("_"*50)
czytaj_liste(nf)
czytaj_slownik(bk)
