public class Bank implements BankVerwaltung, BankVerwaltungOnline{
	private Konto[] konten;
	
	//Erzeugt eine neue Bank ohne Konten
	public Bank(){
		konten = new Konto[0];
	}
	
	//Prueft ob diese Bank das angegebene Konto bereits verwaltet
	public boolean haveKonto(Konto k){
		for (int i=0; i<konten.length; i++){
			if (konten[i]==k) return true;
		}
		return false;
	}
	
	//Fuegt ein neues Konto zur bank hinzu
	public void addKonto(Konto k){
		if (haveKonto(k)) return;
		
		Konto[] konten_neu = new Konto[konten.length + 1];
		for (int i=0; i<konten.length; i++){
			konten_neu[i] = konten[i];
		}
		konten_neu[konten.length] = k;
		
		konten = konten_neu;
	}
	
	//Fuegt ein neues Girokonto zur bank hinzu
	public void addKonto(Girokonto k){
		addKonto((Konto)k);
	}
	
	//entfernt ein bestehendes Konto aus der Bank
	public void removeKonto(Konto k){
		if (!haveKonto(k)) return;
		
		Konto[] konten_neu = new Konto[konten.length - 1];
		int pos = 0;
		
		for (int i=0; i<konten.length; i++){
			if (konten[i]!=k) konten_neu[pos++] = konten[i];
		}
		
		konten = konten_neu;
	}
	
	//Gibt die Anzahl der verwalteten Konten aus
	public int verwalteteKonten(){
		return konten.length;
	}
	
	public String toString(){
		String res = "";
		for (int i=0; i<konten.length; i++){
			if (i>0) res+="\n";
			res += (i+1) + ": " + konten[i].toString();
		}
		
		return res;
	}
	
	public Konto hoechsterSaldo(){
		Comparable c = Maximum.max(konten);
		return (Konto)c;
	}
}