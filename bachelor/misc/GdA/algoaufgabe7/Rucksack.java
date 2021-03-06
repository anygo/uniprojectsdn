import java.util.*;

class Rucksack implements RucksackLoeser {

	int groesse;
	int[] elemente;
	int[][] tabelle;

	Rucksack(String[] args) {
		
		// Rucksackgroesse ermitteln
		try {
			groesse = Integer.parseInt(args[0]);
		} catch (NumberFormatException e) {
			System.out.println("Fehler: Nur Ganzzahlen angeben! - Setze Rucksackgroesse auf 0");
			groesse = 0;
		}

		//Elemente in ein Array "elemente" schreiben
		elemente = new int[args.length - 1];
		for (int i = 0; i < args.length - 1; i++) {
			try {
				elemente[i] = Integer.parseInt(args[i+1]);
			} catch (NumberFormatException e2) {
				System.out.println("Fehler: Nur Ganzzahlen angeben! - Setze Element " + i + " auf 1");
				elemente[i] = 1;
			}
		}

		//Die Elemente noch sortieren...
		Arrays.sort(elemente);
	}

	
	public void solve() {
		
		//2-dimensionales Integer Array erstellen
		tabelle = new int[elemente.length][groesse+1];
		
		for (int i = 0; i < elemente.length; i++) {
			for (int j = 0; j < groesse + 1; j++) {
				if (j == 0) {
					tabelle[i][j] = NICHT_PACKEN;
				//} else if (j == elemente[i]) {
				//	tabelle[i][j] = PACKEN;
				}
				//wenn in der selben Spalte eine Zeile hoeher eine 1 oder 0 steht, dann tabelle[i][j] = 0
				//aber erst ab Zeile 2 (i > 0), da bei Zeile 1 keine "hoehere" Zeile existiert 
				 else if (i > 0 && (tabelle[i-1][j] == PACKEN || tabelle[i-1][j] == NICHT_PACKEN)) {
					tabelle[i][j] = NICHT_PACKEN;

				} else if (j == elemente[i]) {
					tabelle[i][j] = PACKEN;

				} else if (i > 0 && j - elemente[i] > 0 && (tabelle[i-1][j-elemente[i]] == PACKEN || tabelle[i-1][j-elemente[i]]  == NICHT_PACKEN)) {
					tabelle[i][j] = PACKEN;

				} else {
					tabelle[i][j] = UNMOEGLICH;
				}

			}
		}
	}

	
	public void printmatrix() {

		//nur geeignet, falls keine Zahlen groesser 99 vorkommen...
		
		System.out.println();

		//erste Zeile - "Tabellenkopf"
		System.out.print("   ||");
		for (int i = 0; i < groesse + 1; i++) {
			if (i >= 0 && i < 10) {
				System.out.print(" ");
			}
			System.out.print(i + " |");
		}
		System.out.println();
		
		//Trennlinie
		System.out.print("=====");
		for (int i = 0; i <= groesse; i++) {
			System.out.print("====");
		}
		System.out.println();

		//die "eigentliche" Tabelle
		for (int zeile = 0; zeile < elemente.length; zeile++) {
			if (elemente[zeile] >= 0 && elemente[zeile] < 10) {
				System.out.print(" ");
			}
			System.out.print(elemente[zeile] + " ||");
			for (int spalte = 0; spalte <= groesse; spalte++) {
				System.out.print(" ");
				if (tabelle[zeile][spalte] == -1) {
					System.out.print("-" + " |");
				} else {
					System.out.print(tabelle[zeile][spalte] + " |");
				}
			}
			System.out.println();
		}
	}


	public void printsolution() {

		int zielZeile = -1;
		int restGroesse = groesse;

		//In loesung werden die Ergebnisse fuer die spaetere Konsolenausgabe geschrieben
		String loesung = "keine";

		int zeileTmp = elemente.length - 1;

		while (restGroesse != 0) {
			for (int zeile = zeileTmp; zeile >= 0; zeile--) {
				if (tabelle[zeile][restGroesse] == PACKEN) {
					zielZeile = zeile;
					if (loesung.equals("keine")) {
						loesung = "";
					}
					if (!(loesung.equals(""))) {
						loesung += ", ";
					}
					loesung += elemente[zeile];
					restGroesse -= elemente[zeile];

					zeileTmp = zeile - 1;

					break;
				}
			}
			if (zielZeile == -1) {
				System.out.println("Es gibt keine Loesung fuer diese Konstellation von Elementen");
				return;
			}
		}

		//Es folgt nun die Ausgabe
		System.out.println("Fuer einen Rucksack der Groesse " + groesse + " werden folgende Elemente benoetigt: " + loesung);
		System.out.println();
	}

	public static void main(String[] args) {
		
		if (args.length < 2) {
			System.out.println("Start des Programms in folgendem Format:");
			System.out.println("java Rucksack RUCKSACKGROESSE ELEMENTE*");
			return;
		}
			
		int tmp;
		try {
			tmp = Integer.parseInt(args[0]);
		} catch (NumberFormatException e3) {
			System.out.println("Nur Ganzzahlen als Parameter!");
			return;
		}
		if (tmp < 0) {
			System.out.println("Nur positive Ganzzahlen fuer die Rucksackgroesse!");
			return;
		}

		Rucksack rs = new Rucksack(args);

		rs.solve();
		rs.printmatrix();
		System.out.println();
		rs.printsolution();
	}
}
