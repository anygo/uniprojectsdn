/**
 * @author Silvia Schreier<sisaschr@stud.informatik.uni-erlangen.de>
 */
public interface ArraySumme {

	/**
	 * berechnet (parallel) die Summe aller Feld-Eintraege
	 * 
	 * @param array
	 *            die zu summierenden Werte
	 * @param threads
	 *            die zu verwendende Thread-Anzahl
	 * @return die berechnete Summe
	 */
	public long summe(long[] array, int threads);

}
