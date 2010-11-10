interface BankVerwaltung{
	//Prueft ob diese Bank das angegebene Konto bereits verwaltet
	boolean haveKonto(Konto k);
	
	//Fuegt ein neues Konto zur Bank hinzu
	void addKonto(Konto k);
	
	//entfernt ein bestehendes Konto aus der Bank
	void removeKonto(Konto k);
	
	//Gibt die Anzahl der verwalteten Konten aus
	int verwalteteKonten();
	
	//Gibt eine Liste aller verwalteten Konten als Zeichenkette zurueck
	String toString();
	
	//Liefert das Konto mit dem hoechsten Saldo
	Konto hoechsterSaldo();
}