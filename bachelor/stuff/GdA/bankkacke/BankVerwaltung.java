interface BankVerwaltung{
	//Prueft ob diese Bank das angegebene Konto bereits verwaltet
	public boolean haveKonto(Konto k);
	
	//Fuegt ein neues Konto zur Bank hinzu
	public void addKonto(Konto k);
	
	//entfernt ein bestehendes Konto aus der Bank
	public void removeKonto(Konto k);
	
	//Gibt die Anzahl der verwalteten Konten aus
	public int verwalteteKonten();
	
	//Gibt eine Liste aller verwalteten Konten als Zeichenkette zurueck
	public String toString();
}