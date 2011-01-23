
public class Hash {

	/*********************************************************************************************************/

	public Hash(int multiplier_, int modulo_, int offset_) {
		multiplier = multiplier_;
		modulo = modulo_;
		offset = offset_;
		
		buckets = new int[modulo_];
		for (int i = 0; i < modulo_; i++) {
			buckets[i] = -1;
		}
	}
	
	/*********************************************************************************************************/

	public int computeKey(int value) {
		return (value*multiplier)%modulo;
	}
	
	/*********************************************************************************************************/
	
	public String toString() {
		String s = "";
		for (int i = 0; i < modulo; i++) {
			if (buckets[i] == -1) s += "x ";
			else s += buckets[i] + " ";			
		}
		return s;
	}
	
	/*********************************************************************************************************/
		
	/*
	*	Implementieren sie diese Methode. Die Methode soll true zurueckliefern wenn der Wert 'value'
	*	im Hash enthalten ist, andernfalls false. Durchsuchen sie den Hash _nicht_ linear. Denken sie an die
	*	ggf. verwendete Sondierung
	*/
	public boolean isContained(int value) {
		int tmp = buckets[computeKey(value)];
		if (tmp == value)
			return true;
		if (tmp == -1)
			return false;
		int tmp_offset = (tmp + offset) % modulo;
		while (tmp_offset != tmp) {
			if (buckets[tmp_offset] == value)
				return true;
			tmp_offset = (tmp_offset + offset) % modulo;
		}
		return false;
	}

	/*********************************************************************************************************/
	
	/*
	*	Implementieren sie diese Methode. Die Methode fuegt den Wert 'value' in den Hash ein. Wenn der
	*	Wert bereits im Hash enthalten ist soll ("Wert " + value + " schon enthalten") ausgegeben werden.
	*	Falls auch mit Sondierung kein Speicherplatz fuer den Wert gefunden werden kann soll
	*	("Kein Speicher fuer Wert " + value + " gefunden") ausgegeben werden
	*/
	public void insert(int value) {
		int tmp = buckets[computeKey(value)];
		if (tmp == value) {
			System.out.println("Wert " + value + " schon enthalten");
			return;
		}
		if (tmp == -1) {
			buckets[computeKey(value)] = value;
			return;
		}
		int i = (computeKey(value) + offset) % modulo;
		while (i != computeKey(value)) {
			if (buckets[i] == -1) {
				buckets[i] = value;
				return;
			}
			if (buckets[i] == value) {
				System.out.println("Wert " + value + " schon enthalten");
				return;
			}
			i= (i + offset) % modulo;
		}
		System.out.println("Kein Speicher fuer Wert " + value + " gefunden");
	}
	
	/*********************************************************************************************************/
	
	private int[] buckets;
	private int multiplier;
	private int modulo;
	private int offset;			// konstanter Offset fuer die Sondierung

    public static void main(String [] args) {
		Hash h = new Hash(7, 11, 3);
		
		for (int i = 0; i < 14; i++) {
			h.insert(i);
			System.out.println(h);
		}

		for (int i = 0; i < 11; i++) {
			h.insert(i);
			System.out.println(h);
		}

    }
    

}
