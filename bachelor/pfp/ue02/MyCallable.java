import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class MyCallable implements Callable<Integer> {

	private int j;
	private int i;

	public MyCallable(int i, int j) {
		this.i = i;
		this.j = j;
	}

	public Integer call() throws Exception {
		return i + j;
	}

	public static void main(String[] args) {
		ExecutorService e = Executors.newFixedThreadPool(2);
		// Alternative:
		// ExecutorService e = Executors.newCachedThreadPool();

		Future<Integer> f = e.submit(new MyCallable(3, 4));
		try {
			System.out.println(f.get());
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		} catch (ExecutionException e1) {
			e1.printStackTrace();
		}
		e.shutdown();
		try {
			e.awaitTermination(10, TimeUnit.SECONDS);
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		}
	}
}

