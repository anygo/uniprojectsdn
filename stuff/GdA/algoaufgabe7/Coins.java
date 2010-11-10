import java.util.*;


public class Coins {

//***************************************************************************************************

	private int[] coin_types;

//***************************************************************************************************

	/**
		Konstruktor, legt ein neues Array/Reihung fuer die Muenzwerte an
		und traegt die per Kommandozeile uebergebenen Werte ein.
		Wichtig: die einzelnen Muenzwerte stehen in den Eintraegen
		args[0]....args[args.length-2]; das Argument args[args.length-1]
		enthaelt die zu wechselnde Summe und darf nicht in den array
		coin_types eingefuegt werden!
		
		Rufen sie die private Methode 'sortCoinsArray' auf, nachdem sie
		den Array/Reihung mit den Muenzwerten gefuellt haben.

		Vervollstaendigen sie diese Methode
	*/
	Coins(String[] args) {
		coin_types = new int[args.length - 1];
		for (int i = 0; i <= (args.length - 2); i++) {
			int tmp = 0;
			
			try {
				tmp = Integer.parseInt(args[i]);
			} catch (NumberFormatException e2) {
				tmp = 1;	//Sonst kommt es spaeter zu Exceptions
				System.out.println("Fehler: Bitte nur Integer als Kommandozeilenparameter! - stattdessen 1");
			}
			if (tmp == 0) {
				System.out.println("0 als Muenzgroesse nicht zulaessig - stattdessen 1");
				tmp = 1;
			}
			if (tmp < 0) {
				System.out.println("Negative Muenzgroessen? Wenn du meinst ;-)");
			}
			
			coin_types[i] = tmp;
		}

		sortCoinsArray();
	}
	
//***************************************************************************************************

	/**
		Sortiert die in 'coin_types' gespeicherten Muenzwerte.
		Verwenden sie zum Sortieren des Arrays/Reihung 'coin_types'
		die statische Methode 'Arrays.sort()' der Klasse 'Arrays'.
		Um die Klasse Arrays verwenden zu koennen, muss etwas
		importiert werden. Suchen sie dazu im Internet nach der 'Java Class Arrays'

		Vervollstaendigen sie diese Methode
	*/
	private void sortCoinsArray() {
		Arrays.sort(coin_types);
	}
	
//***************************************************************************************************
	
	/**
		Formatierte Ausgabe. Implementieren sie diese Funktion, die durch
		System.out.println() verursachte Ausgabe sollte in etwa wie folgt
		aussehen:
		"Wir haben folgende Muenzen: 1 2 5 10 20 50"

		Vervollstaendigen sie diese Methode
	*/
	public String toString() {
		String ret = "Wir haben folgende Muenzen: ";

		for (int i = 0; i < (coin_types.length); i++) {
			ret += coin_types[i] + " ";
		}

		return ret;
	}

//***************************************************************************************************

	/**
		Findet entsprechend des gierigen Algorithmuses aus
		der Vorlesung die (mehr oder weniger) optimale Zusammensetzung
		der 'summe' aus den bekannten Muenzwerten.
		Geben sie die Loesung in folgender Form aus:
		"97 kann gewechselt werden in: (1*50cent) (2*20cent) (1*5cent) (1*2cent)"

		Vervollstaendigen sie diese Methode
	*/
	public void getCoins(int summe) {
		
		int[] anzahlMuenzen = new int[coin_types.length];
		int summe_new = summe;

		for (int i = coin_types.length - 1; i >= 0; i--) {
			anzahlMuenzen[i] = summe_new / coin_types[i];
			summe_new = summe_new % coin_types[i];
		}
		
		if (summe_new != 0) {
			System.out.println(summe + " kann mit dem angegebenen Muenzset nicht vollstaendig ausgezahlt werden.");
		} else {
			System.out.print(summe + " kann gewechselt werden in:");
			for (int i = anzahlMuenzen.length - 1; i >= 0; i--) {
				if (anzahlMuenzen[i] > 0) {
					System.out.print(" (" + anzahlMuenzen[i] + "*" + coin_types[i] + "cent)");
				}
			}
			System.out.println();
		}
	}


//***************************************************************************************************

	/**
		Programmeinsprungpunkt.
		Hier sollte nichts veraendert werden muessen.
	*/
	public static void main(String[] args) {
	
		if( args.length < 2 ) {
			System.out.println("Syntax Coins:" );
			System.out.println("   Coins [coins]* [amount]");
			System.out.println("   i.e. Coins 1 2 5 10 20 50 263");
			return;
		}

		int summe;
		try {
			summe = Integer.parseInt(args[args.length-1]);
		} catch (NumberFormatException e) {
			System.out.println("Fehler: Das letzte Argument (Betrag) ist keine Ganzzahl");
			System.out.println(e);
			return;
		}

		Coins c = new Coins(args);
		System.out.println(c);
		c.getCoins(summe);

	}

}


