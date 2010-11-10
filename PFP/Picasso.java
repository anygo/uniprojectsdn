import java.util.Random;
  
public class Picasso extends Thread {

	public static void main(String[] args) {
		if (args.length != 3) return;
		
		final int lang = Integer.parseInt(args[0]);
		final int breit = Integer.parseInt(args[1]);
		final int threads = Integer.parseInt(args[2]);
		final Leinwand leinwand = new Leinwand(lang, breit);
		
		for (int i= 0; i < threads; i++) {
			final int bla = i;
			new Thread(new Runnable() {
				public void run() {
					int x, y;
					Random zufall = new Random();
					while (!leinwand.fertig()) {
						x = zufall.nextInt(breit);
						y = zufall.nextInt(lang);
						synchronized (leinwand) {
							if (!leinwand.istGefaerbt(x, y)) {
								leinwand.faerbe(x, y, bla);
								try {
									Thread.sleep(1);
								} catch (Exception e) {
									System.out.println("Fehler");
									return;
								}
						}
						}
					
					}
				}
			}).start();
		}
	}
}
