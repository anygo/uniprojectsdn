interface BankVerwaltungOnline{
	//Prueft ob diese Bank das angegebene Konto bereits verwaltet
	public boolean haveKonto(Konto k);
	
	//Fuegt ein neues Girokonto zur Bank hinzu
	public void addKonto(Girokonto k);
	
	//entfernt ein bestehendes Konto aus der Bank
	public void removeKonto(Konto k);
	
	//Gibt die Anzahl der verwalteten Konten aus
	public int verwalteteKonten();
	
	//Gibt eine Liste aller verwalteten Konten zurueck
	public String toString();
	
	//Liefert das Konto mit dem hoechsten Saldo
	Konto hoechsterSaldo();
}