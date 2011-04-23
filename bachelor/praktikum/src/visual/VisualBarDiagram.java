package visual;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.IOException;

import framed.FrameSource;

/**
 * class extending Frame Visualizer to create bar diagrams
 * @author Matthias Ring, Markus Goetze
 *
 */
public class VisualBarDiagram extends FrameVisualizer {
	private static final long serialVersionUID = 266872307577562915L;

	/**
	 * Constructor in case the source is a buffered audio source, this source must be set later.
	 * @param title Title to be used
	 */
	protected VisualBarDiagram(String title) {
		super(title);
	}

	/**
	 * Constructor in case the source is a stream, the source is set here.
	 * @param source Audio source
	 * @param title Title to be used
	 */
	public VisualBarDiagram(FrameSource source, String title) {
		super(source, title);
	}

	/**
	 * reads data into buffer and normalizes to given range
	 * @param buf Buffer to fill
	 */
	@Override
	public boolean read(double[] buf) throws IOException {
		boolean ret = super.read(buf);
		if (!ret) {
			repaint();
			return false;
		}
		normalizeBuffer();
		repaint();
		return ret;
	}
	
	/**
	 * computes new data and repaints the image
	 */
	@Override
	public void repaint() {
		double nullLine;
		nullLine = (0 - min) / Math.abs(max - min);
		nullLine -= 0.5;
		nullLine *= 2;	
		nullLine = (int) (getHeight() / 2 - (getHeight()/2) * nullLine);
		
		middleOfVals = getHeight() / 2;
		pixelBuffer = new double[frameSize];
		
		// scale to window height
		for (int i = 0; i < frameSize; ++i) {
			pixelBuffer[i] = middleOfVals - originalBuffer[i] * middleOfVals;
		}
		
		super.repaint();
	}

	/**
	 * creates the image and draws it
	 */
	@Override
	public void paint(Graphics g) {
		int w = getWidth() - 2 * indentBorder;
		double inv_skalierung_x = (((double) w) / frameSize);
		
		if (pixelBuffer.length  < 2) return;
		
		image = createImage(getWidth(), getHeight());
		Graphics2D bg = (Graphics2D) image.getGraphics();

		double nullLine;
		nullLine = (0 - min) / Math.abs(max - min);
		nullLine -= 0.5;
		nullLine *= 2;	
		nullLine = (int) (getHeight() / 2 - (getHeight()/2) * nullLine);

		drawBackground(bg, getWidth(), getHeight());
		drawCoordinateSystem(bg, getWidth(), getHeight(),  w, "kHz",
				"", "", getTitle(), (int) nullLine);
		bg.setColor(DRAWING_COLOR);		

		// draw data
		for (int i = 0; i < frameSize; i++) {
			bg.drawLine(((int) (indentBorder + i * inv_skalierung_x)),
					middleOfVals, ((int) (indentBorder + i * inv_skalierung_x)),
					(int) pixelBuffer[i]);
			bg.drawLine(((int) (indentBorder + i * inv_skalierung_x)),
					((int) (pixelBuffer[i])), ((int) (indentBorder + (i + 1)
							* inv_skalierung_x)), ((int) (pixelBuffer[i])));
			bg.drawLine(((int) (indentBorder + (i + 1) * inv_skalierung_x)),
					((int) (pixelBuffer[i])), ((int) (indentBorder + (i + 1)
							* inv_skalierung_x)), middleOfVals);

		}
		// double buffer
		g.drawImage(image, 0, 0, this);
	}
}
