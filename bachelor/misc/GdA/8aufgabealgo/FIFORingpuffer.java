public class FIFORingpuffer implements FIFOSchlange {

	private int[] puffer;
	private int firstindex;
	private int lastindex;
	private boolean isEmpty;
	private int numberElements;

	/**
	* Konstruiert eine neue RingPufferWarteschlange mit 'n' Warteplaetzen.
	*/
	public FIFORingpuffer(int n) {
		if (n <= 0) {
			System.out.println("Fehler: Kapazitaet des Ringpuffers muss > 0 sein.");
			return;
		}
		puffer = new int[n];
		firstindex = 0;
		lastindex = 0;
		isEmpty = true;
		numberElements = 0;
	}

	public boolean einfuegen(int zahl) {
		if (isEmpty == true) {
			puffer[firstindex] = zahl;
			increment_lastindex();
			isEmpty = false;
			numberElements++;
			return true;
		} else if (firstindex == lastindex) {
			System.out.println("Fehler: Ringpuffer bereits voll.");
			return false;
		} else {
			puffer[lastindex] = zahl;
			increment_lastindex();
			numberElements++;
			return true;
		}
	}

	public int herausnehmen() {
		if (isEmpty == true) {
			System.out.println("Fehler: Ringpuffer ist leer");
			return -1;
		} else {
			int tmp = firstindex;
			increment_firstindex();
			numberElements--;
			if (firstindex == lastindex) {
				isEmpty = true;
			}
			return puffer[tmp];
		}
	}

	public String toString() {
			String elemente = new String();
			if (isEmpty == true) {
				elemente += " keine";
			} else {
				int tmp = firstindex;
				for (int i = 0; i < numberElements; i++) {
					elemente += " " + puffer[firstindex];
					increment_firstindex();
				}
				firstindex = tmp;
			}

			return ("Schlange vom Typ Ringpuffer, Kapazitaet: "+puffer.length+", Elemente:"+elemente+", erster Index: "+firstindex+", letzter Index: "+lastindex);
	}

	public int[] eintraege() {
		if (numberElements == 0) {
			return null;
		}
		int[] toReturn = new int[numberElements];
		int tmp = firstindex;
		for (int i = 0; i < toReturn.length; i++) {
			toReturn[i] = herausnehmen();
		}
		firstindex = lastindex = tmp;
		for (int i = 0; i < toReturn.length; i++) {
			einfuegen(toReturn[i]);
		}
		return toReturn;
	}

	private void increment_lastindex() {
		// Methode, die lastindex um 1 "erhoeht", unter Beachtung 
		// der "Ring"-Eigenschaft des Ringpuffers
		if (lastindex < (puffer.length - 1)) {
			lastindex++;
		} else {
			lastindex = (lastindex + 1) % puffer.length;
		}
	}

	private void increment_firstindex() {
		// Methode, die firstindex um 1 "erhoeht", unter Beachtung 
		// der "Ring"-Eigenschaft des Ringpuffers
		if (firstindex < (puffer.length - 1)) {
			firstindex++;
		} else {
			firstindex = (firstindex + 1) % puffer.length;
		}
	}
}
