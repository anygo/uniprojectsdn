public class Banken{
	static void kontenTest(BankVerwaltung bank){
		Konto k1 = new Tagesgeldkonto("K_500");
		Konto k2 = new Tagesgeldkonto("K_10");
		Konto k3 = new Tagesgeldkonto("K_12000");
		k1.einzahlen(500);
		k2.einzahlen(10);
		k3.einzahlen(12000);
		
		bank.addKonto(k1);
		bank.addKonto(k2);
		bank.addKonto(k2);
		bank.addKonto(k3);
	}
	
	static void startOnlinePortal(BankVerwaltungOnline bank){
		Girokonto gk = new Girokonto("Giro", 2000);
		bank.addKonto(gk);
		
		//Tagesgeldkonto tk = new Tagesgeldkonto("Tage");
		//bank.addKonto(tk);
	}
	
	public static void main(String[] args){
		Bank b1 = new Bank();
		
		kontenTest(b1);
		startOnlinePortal(b1);
		
		System.out.println(b1);
		Konto k = b1.hoechsterSaldo();
		
		System.out.println("Bester Kunde: "+k);
	}
}