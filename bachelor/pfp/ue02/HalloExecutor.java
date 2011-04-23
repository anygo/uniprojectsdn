import java.util.concurrent.*;

public class HalloExecutor extends Thread implements Runnable {
	
	private int num;

	public HalloExecutor(int n) {
		this.num = n;
	}

	public void run() {
		System.out.println("Hallo Executor-" + num);
	}
	
	public static void main(String[] args) {
		if (args.length != 1) return;
		int cnt;
		try {
			cnt = Integer.parseInt(args[0]);
		} catch (Exception e) {
			System.err.println("Nur Integer!");
			return;
		}
		if (cnt < 0) {
			System.err.println("Nur positive Werte!");
			return;
		}
		
		ExecutorService ex = Executors.newFixedThreadPool(cnt);

		for (int i = 0; i < cnt; i++) {
			Future<?> f = ex.submit(new HalloExecutor(i));
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}

			}
		
		ex.shutdown();

		try {
			ex.awaitTermination(23, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}
