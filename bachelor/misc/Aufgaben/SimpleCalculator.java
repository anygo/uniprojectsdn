
public class SimpleCalculator {

	public static void main(String args[]) {
		
		if (args.length != 3) {					//Test ob 3 Parameter angegeben wurden
			System.out.println("Fehler!");
			System.out.println("Ungueltige Anzahl von Parametern. Es muessen genau 3 angegeben werden.");
			System.out.println("Aufruf des Programms: java SimpleCalculator ZAHL1 add|sub|mul|div ZAHL2");
			System.out.println("\nBeispiel: java SimpleCalculator 3 add 5");
			
		} else {
			
			double par0 = Double.parseDouble(args[0]);	//Umwandlung des 0. und 2. Parameters nach Double
			double par2 = Double.parseDouble(args[2]);
		
									//Abfrage, welche Rechenoperation verlangt ist
									//und Durchführung der Operation
			if (args[1].equals("add")) {
				System.out.println("Addition: "+par0+" + "+par2+" = "+(par0+par2));
			
			} else if (args[1].equals("sub")) {
				System.out.println("Subtraktion: "+par0+" - "+par2+" = "+(par0-par2));
				
			} else if (args[1].equals("mul")) {
				System.out.println("Multiplikation: "+par0+" * "+par2+" = "+(par0*par2));
			
			} else if (args[1].equals("div")) {
				System.out.println("Division: "+par0+" / "+par2+" = "+(par0/par2));
				
			} else {					//falls keine gültige Operation angegeben: Fehlermeldung
				System.out.println("Fehler!");
				System.out.println("Es stehen nur folgende Rechenoperationen zur Verfügung:");
				System.out.println(" - Addition (add)\n - Subtraktion (sub)\n - Multiplikation (mul)\n - Division (div)");
			}
		}
		
	}
}

