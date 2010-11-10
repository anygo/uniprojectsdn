
public class Fehler {
	///Berechnet den Fehler des aktuellen Ergebnisses
	static float fehler(float wurzelVon, float wurzel){
		return (wurzel*wurzel) - wurzelVon;
	}
	
	///Verbessert das Ergebnis nach dem Heron-Verfahren
	static float heronStep(float wurzelVon, float wurzel, int step){
		System.out.print("    "+step+": "+wurzel +" --> ");
		wurzel = (wurzel + (wurzelVon/wurzel)) / 2;
		System.out.println(wurzel + " (Fehler:" + fehler(wurzelVon, wurzel) + ")");
		return wurzel;
	}
	
	///Gibt true zurueck, wenn das Ergebnis ausreichend gut ist
	static boolean goodSolution(float wurzelVon, float wurzel) {
		if (fehler(wurzelVon, wurzel)< 0.0001) return true;
		return false;
	}
	
	public static void main(String[] args){
		float wurzelVon = 9;
		int step = 0;
		float wurzel = heronStep(wurzelVon, 1, step++);
		
		
		while (!goodSolution(wurzelVon, wurzel))
			wurzel = heronStep(wurzelVon, wurzel, step++);
		
		System.out.println("\nDie Wurzel von "+wurzelVon+" ist "+wurzel+".");
	}
}
