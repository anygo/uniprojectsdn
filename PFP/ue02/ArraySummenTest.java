import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * Kleiner Beispiel JUnit-Test, um die eigene Implementierung zu testen.
 * 
 * @author Silvia Schreier<sisaschr@stud.informatik.uni-erlangen.de>
 */
public class ArraySummenTest {

	private long[] generateRandomArray(final int size) {
		final long[] array = new long[size];
		for (int i = 0; i < size; i++) {
			array[i] = (long) (Math.random() * 1000);
		}
		return array;
	}

	private long summe(final long[] array) {
		long summe = 0;
		for (final long a : array) {
			summe += a;
		}
		return summe;
	}

	/**
	 * vergleicht das parallele Ergebnis mit einer sequentiellen (richtigen)
	 * Implementierung (Array-Groesse 1)
	 */
	@Test
	public void testArraySummeKlein() {
		final int size = 10;
		final int threads = 24;
		final long[] array = generateRandomArray(size);
		final ArraySumme sum = new ArraySummeImpl();
		final long result = sum.summe(array, threads);
		final long seqResult = summe(array);
		assertEquals(seqResult, result);
	}

	/**
	 * vergleicht das parallele Ergebnis mit einer sequentiellen (richtigen)
	 * Implementierung bei Arrays mittlerer Groesse und relativ wenigen Threads
	 */
	@Test
	public void testArraySummeMittel() {
		for (int threads = 2; threads < 15; threads++) {
			for (int size = 500; size < 5000; size += 355) {
				final long[] array = generateRandomArray(size);
				final ArraySumme sum = new ArraySummeImpl();
				final long result = sum.summe(array, threads);
				final long seqResult = summe(array);
				assertEquals(seqResult, result);			
			}
		}
	}

	/**
	 * vergleicht das parallele Ergebnis mit einer sequentiellen (richtigen)
	 * Implementierung bei vielen Threads und grossem Array
	 */
	@Test
	public void testArraySummeGross() {
		final int size = 1000000;
		final int threads = 123;
		final long[] array = generateRandomArray(size);
		final ArraySumme sum = new ArraySummeImpl();
		final long result = sum.summe(array, threads);
		final long seqResult = summe(array);
		assertEquals(seqResult, result);
	}

	/**
	 * vergleicht das parallele Ergebnis mit einer sequentiellen (richtigen)
	 * Implementierung (Array-Groesse 0)
	 */
	@Test
	public void testArraySummeLeer() {
		final int size = 0;
		final int threads = 24;
		final long[] array = generateRandomArray(size);
		final ArraySumme sum = new ArraySummeImpl();
		final long result = sum.summe(array, threads);
		final long seqResult = summe(array);
		assertEquals(seqResult, result);
	}
}
