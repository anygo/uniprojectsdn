package visual;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Locale;

import javax.imageio.ImageIO;
import javax.swing.JComponent;

import framed.FrameSource;

/**
 * Abstract class for drawing diagrams based on frames. For now, there diagrams may either be functions or bar diagrams.
 * @author Matthias Ring, Markus Goetze
 *
 */
public abstract class FrameVisualizer extends JComponent implements FrameSource  {

	protected static final long serialVersionUID = -1L;
	
	/**
	 * data source to fill originalBuffer
	 */
	protected FrameSource frameSource = null;
	
	/**
	 * size of originalBuffer, size of frame
	 */
	protected int frameSize = 0;
	
	/**
	 * buffer containing original data
	 */
	protected double[] originalBuffer = null;
	
	/**
	 * buffer containing values to be displayed (as pixels)
	 */
	protected double[] pixelBuffer = new double[frameSize];
	
	/**
	 * half the range of values currently being displayed
	 */
	protected int middleOfVals;
	
	/**
	 * min and max value on y-axis
	 */
	protected double max = Double.MAX_VALUE;
	protected double min = Double.MIN_VALUE;
	
	/**
	 * global min and max
	 */
	private double globalMax = Double.MIN_VALUE;
	private double globalMin = Double.MAX_VALUE;
	
	/**
	 * Default component size
	 */
	protected static Dimension DEFAULT_SIZE = new Dimension(524, 256);
	
	/**
	 * space left blank on left and right (border)
	 */
	protected int indentBorder = 5;
	
	/**
	 * image to be drawn without additional interactive mouse events
	 */
	protected Image image = createImage(getWidth(), getHeight());
	
	/**
	 * title
	 */
	private String title = null;
	
	/**
	 * window type applied for FFT, in case input is not a stream
	 */
	public enum WINDOW_TYPE {
		HAMMING, HANNING, RECTANGLE
	};
	
	/**
	 * Colors for painting
	 */
	public Color TITLE_COLOR = new Color(210, 38, 38);
	public Color AXES_LABEL_COLOR = new Color(42, 181, 5);
	public Color SELECTION_COLOR = new Color(255, 255, 0, 120);
	public Color BACKGROUND_COLOR = Color.WHITE;
	public Color COORDINATE_SYSTEM_COLOR = Color.GRAY;
	public Color DRAWING_COLOR = Color.BLACK;
	public Color CROSSHAIR_COLOR = Color.LIGHT_GRAY;

	/**
	 * constructor for case that source is a buffered audio source, not a stream.
	 * in case of a file source, additional parameters must be set later
	 * @param title title to be used
	 */
	protected FrameVisualizer(String title) {
		this.setTitle(title);
		setPreferredSize(DEFAULT_SIZE);		
	}
		
	/**
	 * constructor for case that source is a stream, not a buffered audio source
	 * in this case the source is known at this point, so it can be set here
	 * @param source frame source
	 * @param title title to be used
	 */
	public FrameVisualizer(FrameSource source, String title) {
		this.setTitle(title);
		this.frameSource = source;
		setPreferredSize(DEFAULT_SIZE);		
		
		adjustToBufferedSource();
	}
	
	/**
	 * in case that source is a buffered audio source, buffer features contuniuosly change and are being adjusted
	 */
	protected void adjustToBufferedSource() {
		frameSize = frameSource.getFrameSize();
		originalBuffer = new double[frameSize];
		setCursor(new Cursor(Cursor.CROSSHAIR_CURSOR));
	}

	/**
	 * sets range of values on y-axis
	 * 
	 * values have to be symmetric
	 * 
	 * @param min
	 * @param max
	 */
	public void setMinMax(double min, double max) {
		this.max = max;
		this.min = min;
	}

	/**
	 * 
	 * @return maximum range of values on y-axis
	 */
	public double getMax() {
		return max;
	}

	/**
	 * 
	 * @return minimum range of values on y-axis
	 */
	public double getMin() {
		return min;
	}
	
	/**
	 * sets the title to be used
	 * @param title
	 */
	protected void setTitle(String title) {
		this.title = title;
	}

	/**
	 * @return title to be used
	 */
	protected String getTitle() {
		return title;
	}

	/**
	 * @return frame size
	 */
	public int getFrameSize() {
		return frameSize;
	}

	/**
	 * fills buffer from source
	 * @return success or failure of read
	 */
	@Override
	public boolean read(double[] buf) throws IOException {
		if (!frameSource.read(buf)) {
			return false;
		}
		System.arraycopy(buf, 0, this.originalBuffer, 0, frameSize);

		return true;
	}
	
	/**
	 * normalizes data to range between min and max
	 */
	
	protected void normalizeBuffer() {			
		for (int i = 0; i < originalBuffer.length; i++) {
			if (originalBuffer[i] > getGlobalMax()) {
				globalMax = originalBuffer[i];
			} else if (originalBuffer[i] < getGlobalMin()) {
				globalMin = originalBuffer[i];
			}
			originalBuffer[i] = (originalBuffer[i] - min) / Math.abs(max - min);
			originalBuffer[i] -= 0.5;
			originalBuffer[i] *= 2;	
		}
	}

	/**
	 * clears image before new draw
	 * @param bg	graphics on which to be drawn
	 * @param width	width of image	
	 * @param height	height of image
	 */
	protected void drawBackground(Graphics bg, int width, int height) {
		bg.setColor(BACKGROUND_COLOR);
		bg.fillRect(0, 0, width, height);
	}

	/**
	 * draws a coordinate system including the axes and their labels and printing the title into the diagram
	 * @param bg graphics on which to be drawn
	 * @param width	width of image
	 * @param height	height of image
	 * @param innerWidth	width of image without border on left and right
	 * @param xLabelRight	x axis label at maximum x position
	 * @param xLabelLeft x axis label at minimum x position
	 * @param yLabel y axis label
	 * @param title title to be used
	 * @param nullPos position of 0 and x-axis on y-axis
	 */
	protected void drawCoordinateSystem(Graphics bg, int width, int height,
			int innerWidth, String xLabelRight, String xLabelLeft, String yLabel, String title, int nullPos) {
		// claculate amount of pixels of title
		FontMetrics fm = bg.getFontMetrics();

		bg.setColor(COORDINATE_SYSTEM_COLOR);
		bg.drawLine(0, nullPos, innerWidth + indentBorder, nullPos);
		bg.drawLine(indentBorder, 5, indentBorder, height - 5);
		bg.setColor(AXES_LABEL_COLOR);
		bg.drawString(yLabel, indentBorder + 5, 17);
		bg.drawString(xLabelRight, innerWidth - fm.stringWidth(xLabelRight), nullPos + 15);
		bg.drawString(xLabelLeft, indentBorder + 5, nullPos + 15);
		
		bg.drawString(roundTwoDecimals(getMin()) + "", indentBorder + 5, getHeight() - indentBorder);
		bg.drawString(roundTwoDecimals(getMax()) + "", indentBorder + 5, 0 + indentBorder + 10);

		bg.setColor(TITLE_COLOR);
		bg.drawString(title, innerWidth - 35 - fm.stringWidth(title), 17);
		
		bg.setColor(COORDINATE_SYSTEM_COLOR);
		int octX = innerWidth / 8;
		int octY = height / 8;

		for (int i = 0; i <= innerWidth; i += octX) {
			bg.drawLine(indentBorder + i, nullPos + 3, indentBorder + i,
					nullPos - 3);
		}

		int cur = nullPos;
		while (cur >= indentBorder) {
			bg.drawLine(indentBorder - 3, cur, indentBorder + 3,
			cur);
			cur -= octY;
		}
		cur = nullPos;
		while (cur <= getHeight() - indentBorder) {
			bg.drawLine(indentBorder - 3, cur, indentBorder + 3,
			cur);
			cur += octY;
		}
//		for (int i = 0; i <= ((height / 2) - indentBorder); i += octY) {
//			bg.drawLine(indentBorder - 3, middleOfVals + i, indentBorder + 3,
//					middleOfVals + i);
//			bg.drawLine(indentBorder - 3, middleOfVals - i, indentBorder + 3,
//					middleOfVals - i);
//		}
	}

	/**
	 * rounds a decimal to 2 decimals
	 * @param d
	 * @return rounded d
	 */
	protected double roundTwoDecimals(double d) {	
		DecimalFormat twoDForm = (DecimalFormat) DecimalFormat.getInstance(Locale.US);
		twoDForm.applyPattern("#.##");
		return Double.valueOf(twoDForm.format(d));
	}
	
	/**
	 * prints image to file
	 * @param file file to printed
	 * @param type type in which image will be saved
	 * @throws IOException
	 */
	public void printToFile(String file, String type) throws IOException {
		ImageIO.write((RenderedImage) image, type, new File(file));
	}

	/**
	 * 
	 * @return the global maximum in this frame
	 */
	public double getGlobalMax() {
		return globalMax;
	}

	/**
	 * 
	 * @return the global minimum in this frames
	 */
	public double getGlobalMin() {
		return globalMin;
	}
}
