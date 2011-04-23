public class Olympia extends Thread {

	public static void main(String[] args) {
		//ExecutorService e = Executors.newFixedThreadPool(args.length);
		final Wettkampf w = new Wettkampf(args);
		final Teilnehmer[] t = w.getTeilnehmer();
		for (int i = 0; i < t.length; i++) {
			final int laeufer = i;
			new Thread(new Runnable() {
				public void run() {
					while (!w.istBeendet()) {
						t[laeufer].laufe();
					}
				}
			}).start();
		}
	}

}
