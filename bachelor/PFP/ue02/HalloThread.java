
public class HalloThread extends Thread {
	
	private int num;

	public HalloThread(int n) {
		this.num = n;
	}

	@Override
	public void run() {
		System.out.println("Hallo Thread-" + num);
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
			HalloThread t = new HalloThread(i);
			t.start();

			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}
