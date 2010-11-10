/**
 * created 09.04.2008
 */
import java.awt.Cursor;

import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 * @author Marc Woerlein <woerlein@informatik.uni-erlangen.de>
 * @author Silvia Schreier <sisaschr@stud.informatik.uni-erlangen.de>
 */
public class Wettkampf {

	private boolean beendet = false;
	final Teilnehmer[] teilnehmer;

	/**
	 * erzeugt einen neuen Wettkampf mit den gegebenen Namen der Teilnehmer
	 * @param namen die Namen der Teilnehmer
	 */
	public Wettkampf(final String[] namen) {
		teilnehmer = new Teilnehmer[namen.length];
		try {
			synchronized (teilnehmer) {
				javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
					public void run() {
						// erzeuge Fenster
						final JFrame mainFrame = new JFrame();
						final JPanel mainPanel = new JPanel();
						mainPanel.setLayout(new BoxLayout(mainPanel,
								BoxLayout.PAGE_AXIS));
						mainFrame.setContentPane(mainPanel);
						mainFrame.setTitle("Wettkampf");
						mainFrame
								.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

						// erzeuge Teilnehmer
						for (int i = 0; i < teilnehmer.length; i++) {
							teilnehmer[i] = new Teilnehmer(Wettkampf.this,
									namen[i], mainFrame);
							mainPanel.add(teilnehmer[i].getPanel());
						}

						// stelle Fenster dar
						mainFrame.pack();
						mainFrame.setLocationRelativeTo(null);
						mainFrame.setCursor(Cursor
								.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
						mainFrame.setVisible(true);
					}
				});
			}
		} catch (final Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * liefert ein Feld aller Teilnehmer des Wettkampfes
	 * @return die Teilnehmer
	 */
	public Teilnehmer[] getTeilnehmer() {
		synchronized (teilnehmer) {
			return teilnehmer;
		}
	}

	/**
	 * ueberprueft ob der Wettkampf beendet ist, also ein Teilnehmer im Ziel ist
	 * @return <code>true</code> wenn ein Teilnehmer das Ziel erreicht hat, ansonsten <code>false</code>
	 */
	public synchronized boolean istBeendet() {
		return beendet;
	}

	/**
	 * benachrichtigt den Wettkampf, dass ein Teilnehmer das Ziel erreicht hat
	 */
	public synchronized void setzBeendet() {
		beendet = true;
	}
}