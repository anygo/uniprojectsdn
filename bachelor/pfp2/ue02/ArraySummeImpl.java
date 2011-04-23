import java.util.concurrent.*;

public class ArraySummeImpl extends Thread implements ArraySumme {

	private int start;
	private int end;
	private long[] arr;

	public ArraySummeImpl(int start, int end, long[] arr) {
		this.start = start;
		this.end = end;
		this.arr = arr;
	}
	
	public Long call() throws Exception {
		long toReturn = 0;
		for (int i = start; i < end; i++) {
			toReturn += arr[i];
		}
		return toReturn;
	}

	public long summe(long[] array, int threads) {
		int step = array.length / threads;
		ExecutorService ex = Executors.newFixedThreadPool(threads);
		Future[] f = new Future[threads];
		
		int i = 0;
		int start = 0;
		while (start < array.length) {
			f[i] = ex.submit(new ArraySummeImpl(start, start+step, array));
			start += step;
			i++;
		}

		long toReturn = 0;
		for (int j = 0; j < threads; j++) {
			toReturn += f[j];
		}

		return toReturn;
	}

}
