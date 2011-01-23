public interface Iterator{
	// Gibt es noch ein naechstes Objekt? 
	boolean hasNext(); 
	
	// Liefert das Object 
	List.Node next();
	
	// Setzt den Iterator an den Anfang zurueck
	void reset();
}