interface FIFOSchlange
{

	/**
	* Fuegt ein Element in die Warteschlange ein.
	* Falls das Element erfolgreich eingefuegt wurde muss 'true' zurueckgegeben werden, andernfalls 'false'
	* Wenn kein Element eingefuegt werden konnte (z.B. weil der Ringpuffer voll ist) soll zusaetzlich eine Fehlermeldung ausgegeben werden.
	*/
	public boolean einfuegen(int zahl);

	/**
	* Nimmt das erste Element (also jenes, welches als erstes eingefuegt wurde) aus der Liste heraus und gibt es zurueck.
	* Wenn die Schlange keine Elemente enthaelt soll ein Fehler ausgegeben werden und -1 zurueckgeben werden.
	*/
	public int herausnehmen();

	/**
	* Gibt einen String mit dem Warteschlangeninhalt sowie weiterfuehrenden Informationen zurueck, z.B.
	* "Schlage vom Typ Ringpuffer, Kapazitaet=10, Elemente: 1 3 4 6 3 2 erster index=3 letzter index=9"
	* "Schlage vom Typ verkettete Liste, 6 Elemente: 1 3 4 6 3 2"
	*/
	public String toString();

	/**
	* Liefert ein Array der gespeicherten Elemente zurueck. Das Array enthaelt
	* alle in der Liste gespeicherten Elemente in der Reihenfolge in der sie aus der FIFO-Warteschlange
	* ausgelesen werden. D.h. das erste Element des zurueckgegebenen Arrays ist das Element, dass
	* beim ersten Aufruf von herausnehmen() zurueckgegeben wuerde. Verwenden Sie bei der Umsetzung
	* dieser Methode nur einfuegen() und herausnehmen(). Der Inhalt der Warteschlange darf durch den Aufruf
	* von eintraege() nicht veraendert werden!
	*/
	public int[] eintraege();


}
