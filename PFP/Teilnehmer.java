/**
 * created 30.03.2008
 */
import java.awt.FlowLayout;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;

/**
 * @author Marc Woerlein<woerlein@informatik.uni-erlangen.de>
 * @author Silvia Schreier<sisaschr@stud.informatik.uni-erlangen.de>
 */
public class Teilnehmer {

	/**
	 * 
	 */
	private final Runnable laeufer;
	final JProgressBar bar;
	private final JPanel panel;

	/**
	 * erzeugt einen neuen Teilnehmer fuer den gegebenen Wettkampf
	 * @param wettkampf der Wettkampf
	 * @param name der Name des Teilnehmers
	 * @param mainFrame der Frame des Wettkampfes
	 */
	Teilnehmer(final Wettkampf wettkampf, final String name,
			final JFrame mainFrame) {
		this.panel = new JPanel();

		// erzeuge Komponenten
		this.bar = new JProgressBar();
		final JLabel label = new JLabel(name + ": ");
		label.setLabelFor(bar); // Barrierefreiheit

		// kombiniere Komponenten
		this.panel.setLayout(new FlowLayout(FlowLayout.RIGHT));
		this.panel.add(label);
		this.panel.add(bar);

		// erzeuge runnable
		this.laeufer = new Runnable() {
			public void run() {
				if (!wettkampf.istBeendet()) {
					// wird nur erhoeht, wenn nicht fertig, also danach immer <=
					// maxium
					bar.setValue(bar.getValue() + 1);
					if (bar.getValue() == bar.getMaximum()) {
						wettkampf.setzBeendet();
						JOptionPane.showMessageDialog(mainFrame, name
								+ " hat gewonnen!", "Ziel",
								JOptionPane.INFORMATION_MESSAGE);
					}
				}
			}
		};
	}

	/**
	 * liefert das zum Teilnehmer gehoerige Panel
	 * @return das Panel
	 */
	JPanel getPanel() {
		return panel;
	}

	/**
	 * der Laeufer bewegt sich einen Schritt nach vorne
	 */
	public void laufe() {
		try {
			// benachrichtige GUI
			javax.swing.SwingUtilities.invokeLater(laeufer);
			// damit man was sieht
			// Thread.sleep(100);

			// um es interresanter zu machen
			Thread.sleep((int) (300 * Math.random()));
		} catch (final InterruptedException e) {
			e.printStackTrace();
		}
	}
}