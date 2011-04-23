/**
 * created 30.03.2008
 */
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 * Erzeugt ein thread-sicheres pixel-basiertes Malfenster
 * 
 * @author Marc Woerlein<woerlein@informatik.uni-erlangen.de>
 */
public class Leinwand {

	private final Color[] palette;

	private final Color[][] feld;
	private int zaehler;
	/* final */Runnable repaint;

	/**
	 * Konstruktor fuer Piccasso (Aufgabenblatt 3)
	 * 
	 * @param x
	 *            Anzahl der Farbflaechen in x-Richtung
	 * @param y
	 *            Anzahl der Farbflaechen in y-Richtung
	 */
	public Leinwand(final int x, final int y) {
		this(x, y, 10, "Leinwand",
				new Color[] { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
						Color.GRAY, Color.PINK, Color.CYAN, Color.DARK_GRAY,
						Color.MAGENTA, Color.ORANGE, Color.LIGHT_GRAY });
	}

	/**
	 * Konstruktor fuer ein beliebiges Pixelfeld
	 * 
	 * @param x
	 *            Anzahl der Farbflaechen in x-Richtung
	 * @param y
	 *            Anzahl der Farbflaechen in y-Richtung
	 * @param pSize
	 *            Seitenlaenge einer Farbflaeche in Pixel
	 * @param titel
	 *            Fensterueberschrift
	 * @param palette
	 *            die zum faerben benutzen Farben
	 */
	public Leinwand(final int x, final int y, final double pSize,
			final String titel, final Color[] palette) {
		this(x, y, pSize, titel, new Color[x][y], palette, x * y);
	}

	/** Konstruktor zur Fenstererzeugung */
	private Leinwand(final int x, final int y, final double pSize,
			final String titel, final Color[][] feld, final Color[] palette,
			final int zaehler) {
		this.feld = feld;
		this.zaehler = zaehler;
		this.palette = palette;
		try {
			javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
				public void run() {
					// erzeuge Fenster
					final JFrame mainFrame = new JFrame();

					// erzeuge Panel, welches das Feld als farbige Flaechen
					// darstellt
					final JPanel colorPanel = new JPanel() {
						private static final long serialVersionUID = 1L;

						@Override
						protected void paintComponent(Graphics g) {
							super.paintComponent(g);

							final double xFac = ((double) getSize().width) / x;
							final double yFac = ((double) getSize().height) / y;
							for (int i = 0; i < x; i++) {
								for (int j = 0; j < y; j++) {
									boolean unset = true;
									synchronized (feld) {
										unset = (feld[i][j] == null);
										g.setColor(unset ? Color.WHITE
												: feld[i][j]);
									}
									// zeichne Flaeche
									g.fillPolygon(new int[] { (int) (i * xFac),
											(int) (i * xFac),
											(int) ((i + 1) * xFac),
											(int) ((i + 1) * xFac) },
											new int[] { (int) (j * yFac),
													(int) ((j + 1) * yFac),
													(int) ((j + 1) * yFac),
													(int) (j * yFac) }, 4);
									if (unset) {
										// zeichne Kreuz
										g.setColor(Color.BLACK);
										g.fillPolygon(new int[] {
												(int) (i * xFac),
												(int) (i * xFac),
												(int) ((i + 1) * xFac),
												(int) ((i + 1) * xFac) },
												new int[] {
														(int) ((j + 1) * yFac),
														(int) (j * yFac),
														(int) ((j + 1) * yFac),
														(int) (j * yFac) }, 4);
									}
								}
							}
						}
					};
					colorPanel.setPreferredSize(new Dimension(
							(int) (pSize * x), (int) (pSize * y)));
					mainFrame.setContentPane(colorPanel);
					mainFrame.setTitle(titel);
					mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

					// stelle Fenster dar
					mainFrame.pack();
					mainFrame.setLocationRelativeTo(null);
					mainFrame.setCursor(Cursor
							.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
					mainFrame.setVisible(true);

					// Nachricht zum neu zeichnen des Fensterns, wenn ein
					// weiteres Feld gesetzt wurde
					repaint = new Runnable() {
						public void run() {
							mainFrame.repaint();
						}
					};
				}
			});
		} catch (final Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Setzt den Punkt (x,y) auf die entsprechende Farbe aus der Farbpalette
	 * 
	 * @param x
	 * @param y
	 * @param farbe
	 */
	public void faerbe(final int x, final int y, final int farbe) {
		// fuer haeufigere Threadwechsel
		Thread.yield();
		synchronized (feld) {
			feld[x][y] = palette[farbe % palette.length];
			zaehler--;
		}
		javax.swing.SwingUtilities.invokeLater(repaint);
	}

	/**
	 * @return <code>true</code>, wenn faerbe, so oft aufgerufen wurde wie es
	 *         Punkte gibt.
	 */
	public boolean fertig() {
		synchronized (feld) {
			return zaehler <= 0;
		}
	}

	/**
	 * ueberprueft, ob ein Punkt schon gefaerbt ist
	 * 
	 * @param x
	 * @param y
	 * @return <code>true</code>, wenn dem Punkt (x,y) schon ein Farbwert
	 *         zugewiesen ist
	 */
	public boolean istGefaerbt(final int x, final int y) {
		synchronized (feld) {
			return feld[x][y] != null;
		}
	}

	/**
	 * @return Anzahl der Punkte in x-Richtung
	 */
	public int getX() {
		synchronized (feld) {
			return feld.length;
		}
	}

	/**
	 * @return Anzahl der Punkte in y-Richtung
	 */
	public int getY() {
		synchronized (feld) {
			return feld.length > 0 ? feld[0].length : 0;
		}
	}

	/**
	 * zum Darstellen eines Pixelbildes in einem seperaten Fenster
	 * 
	 * @param feld
	 *            die einzelnen Pixel des Bildes
	 * @param titel
	 *            Titel des neuen Fensters
	 */
	public static void show(final Color[][] feld, final String titel) {
		new Leinwand(feld.length, feld[0].length, 1, titel, feld, null, 0);
	}

}