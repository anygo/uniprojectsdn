
public class MinimumSuche {

	public static void main(String[] args) {
	
		if (args.length != 3) {					//Test ob genau 3 Parameter angegeben wurden
			System.out.println("Fehler!");
			System.out.println("Bitte genau 3 Zahlen als Parameter angeben!");
			System.out.println("Das Programm ist hiermit beendet.\n");
		
		} else {
		
			double par0 = Double.parseDouble(args[0]);	//Umwandlung der Parameter nach Double
			double par1 = Double.parseDouble(args[1]);
			double par2 = Double.parseDouble(args[2]);
									//Ausgabe mittels println
			System.out.println("Zahl 1: "+par0+" Zahl 2: "+par1+" Zahl 3: "+par2);
	
									//Ausgabe des Minimums durch Aufruf
									//der Methode minSuche
			System.out.println("Minimum: "+minSuche(par0, par1, par2));	
			
		}
		
	}
	
	public static double minSuche(double x, double y, double z) {
		//Methode zur Bestimmung des Minimums der drei Zahlen
		
		if (x <= y && x <= z) return x;					
		else if (y <= x && y <= z) return y;
		else return z;
		
	}	
	
}

