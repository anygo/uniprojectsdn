import java.util.Random;

public class Picasso {

	public static void main(String[] args) {
		if (args.length == 3) {
			final int breite = Integer.parseInt(args[0]);
			final int hoehe = Integer.parseInt(args[1]);
			final int threads = Integer.parseInt(args[2]);
			
			final Leinwand leinwand = new Leinwand(breite, hoehe);
			
			for (int i = 0; i < threads; i++) {
				final int me = i;
				new Thread(new Runnable() {
					public void run() {
						int x,y;
						Random random = new Random();
						
						while (!leinwand.fertig()) {
							int bar = random.nextInt(100); // 0 <= bar < 100
							x = random.nextInt(breite);
							y = random.nextInt(hoehe);
							if (!leinwand.istGefaerbt(x, y)) {
								leinwand.faerbe(x, y, me);
							}
						}
					}
				}).start();
			}
		}
	}

}
