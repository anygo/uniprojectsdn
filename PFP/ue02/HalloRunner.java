import java.util.concurrent.*;

public class HalloRunner implements Runnable {
	
	private int num;

	public HalloRunner(int n) {
		this.num = n;
	}

	public void run() {
		System.out.println("Hallo Runner-" + num);
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
		
		for (int i = 0; i < cnt; i++) {
			Thread t = new Thread(new HalloRunner(i));
			t.start();

			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}
