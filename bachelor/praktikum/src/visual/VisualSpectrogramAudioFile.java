package visual;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;
import exceptions.MalformedParameterStringException;
import framed.*;

/**
 * VisualSpectrogramAudioFile implements a visualization of the spectrogram 
 * processing a given buffered audio file
 * 
 * @author sidoneum, sifeluga
 * 
 */

public class VisualSpectrogramAudioFile extends VisualSpectrogram implements 
															MouseListener, 
															MouseMotionListener, 
															MouseWheelListener, 
															Visualizer,
															ComponentListener,
															FrameChanger {
	
	
	private static final long serialVersionUID = 1L;
	
	/**
	 * list of FrameListeners for communication with spectrum visualizers
	 */
	protected ArrayList<FrameListener> listenerList = new ArrayList<FrameListener>();

	protected Graphics2D g2d;
	
	// indicators
	/**
	 * current horizontal position of the mouse cursor with respect to the component
	 */
	protected int mouseX;
	
	/**
	 * current vertical position of the mouse cursor with respect to the component
	 */
	protected int mouseY;
	
	/**
	 * indicates if the mouse cursor is inside the component
	 */
	protected boolean mouseInside;
	
	/**
	 * indicator for paint-method, whether to show markers or not
	 */
	protected boolean showMarker =  false;
	
	/**
	 * set of variables and indicators for painting the marker-rectangle
	 */
	protected boolean markerDragging = false;
	protected double startMarkerRatio = 0.;
	protected double endMarkerRatio = 0.;
	protected int markX1;
	protected int markX2;
	protected Color markerColor = new Color(255, 255, 0, 100);
	protected Color markerBorderColor = new Color(0, 0, 0, 255);
	
	/**
	 * audio file storage object (reference)
	 */
	protected BufferedAudioSource bas;
	protected int audioFileSize = 0;
	
	/**
	 * current range inside bigSpectrogramBuffer
	 */
	protected int viewFromSample = 0;
	protected int viewNSamples = 0;
	
	/**
	 * buffer for spectrogram data
	 */
	protected double[][] bigSpectrogramBuffer = null;
	protected int bigFromSample = 0;
	protected int bigNSamples = 0;
	
	/**
	 * use average "interpolation"?
	 */
	protected boolean averaged;
	
	/**
	 * set of variables defining the current window
	 */
	protected Window window;
	protected int windowLength;
	protected int windowShift;
	protected String windowType;
	
	/**
	 * return value of window.getShift() is in ms, we need samples
	 */
	protected int samplesPerShift;
	
	
	
///////////////////////////////////////////////////////////////////////////////
///								constructors								///
///////////////////////////////////////////////////////////////////////////////
	
	/**
	 * simple easy-to-use constructor
	 * @param bas reference to global audio file storage
	 */
	public VisualSpectrogramAudioFile(BufferedAudioSource bas) {
		
		this(bas, 10., 0, bas.getSampleRate()/1000 * 25, true, false, false, new Dimension(512, 256), 25, 10, "hamm", false);	
	}
	
	/**
	 * constructor
	 * @param bas reference to global audio file storage
	 * @param fromSample the first sample, starting point
	 * @param nSamples how many samples should be read
	 * @param startComponentSize startSize of JComponent (setPreferredSize(..))
	 */
	public VisualSpectrogramAudioFile(BufferedAudioSource bas, int fromSample, int nSamples, Dimension startComponentSize) {
		
		this(bas, 10., fromSample, nSamples, true, false, false, startComponentSize, 25, 10, "hamm", false);	
	}
	
	/**
	 * constructor
	 * @param bas reference to global audio file storage
	 * @param fromSample the first sample, starting point
	 * @param nSamples how many samples should be read
	 * @param windowLength length of the window in ms
	 * @param windowShift shift of the window in ms
	 */
	public VisualSpectrogramAudioFile(BufferedAudioSource bas, int fromSample, int nSamples, int windowLength, int windowShift) {
		
		this(bas, 10., fromSample, nSamples, true, false, false, new Dimension(512, 256), windowLength, windowShift, "hamm", false);	
	}
	
	/**
	 * constructor
	 * @param bas reference to global audio file storage
	 * @param contrast
	 * @param fromSample the first sample, starting point
	 * @param nSamples how many samples should be read
	 * @param useDefaultMouseListeners use built-in mouse-listeners (zooming, dragging, etc. try out!)
	 * @param colored start with a colored spectrogram
	 * @param log use logScale
	 * @param componentSize startSize of JComponent (setPreferredSize(..))
	 * @param windowLength length of the window in ms
	 * @param windowShift shift of the window in ms
	 * @param windowType hamm|hann|rect
	 * @param averaged use average "interpolation"
	 */
	public VisualSpectrogramAudioFile(BufferedAudioSource bas, 
										double contrast, 
										int fromSample, 
										int nSamples, 
										boolean useDefaultMouseListeners, 
										boolean colored, 
										boolean log, 
										Dimension componentSize,
										int windowLength,
										int windowShift,
										String windowType,
										boolean averaged) {
		
		this.bas = bas;
		this.colored = colored;
		this.log = log;
		this.samplesPerShift = ((bas.getSampleRate() * windowShift) / 1000);
		this.audioFileSize = bas.getBufferSize();

		image = new BufferedImage(componentSize.width, componentSize.height, BufferedImage.TYPE_INT_ARGB);
		setPreferredSize(componentSize);
		
		setWindowConfig(windowType, windowLength, windowShift);
		setContrast(contrast);
		
		setDataRange(fromSample, nSamples);
		
		if (useDefaultMouseListeners) {
			addComponentListener(this);
			addMouseListener(this);
			addMouseMotionListener(this);
			addMouseWheelListener(this);
		}
	}
	
	
///////////////////////////////////////////////////////////////////////////////
///							set- and get-methods							///
///////////////////////////////////////////////////////////////////////////////

	/**
	 * set colors for overlay, etc.
	 */
	public void setColorScheme(Color textColor, Color markerColor, Color markerBorderColor, Color scaleColor) {
		
		this.textColor = textColor;
		this.markerColor = markerColor;
		this.markerBorderColor = markerBorderColor;
		this.scaleColor = scaleColor;
	}
	
	/**
	 * show markers|marker-range
	 * @param show indicator, whether to show markers or not
	 */
	public void setShowMarkers(boolean show) {
		
		this.showMarker = show;
	}
	
	/**
	 * indicates, whether the markers are currently shown
	 * @return marker currently shown?
	 */
	public boolean getShowMarkers() {
		
		return showMarker;
	}
	
	/**
	 * indicates, whether average "interpolation" is used or not
	 * @return average interpolation used?
	 */
	public boolean useAverage() {
		
		return averaged;
	}
	
	/**
	 * allows you to set average "interpolation"
	 * @param averaged avereged or not?
	 */
	public void useAverage(boolean averaged) {
		
		this.averaged = averaged;
	}
	
	/**
	 * sets te current marker range (normalized to complete audio file length)
	 * @param from start
	 * @param to end
	 */
	public void setMarkerRange(double from, double to) {
		
		this.startMarkerRatio = from;
		this.endMarkerRatio = to;
	}
	
	/**
	 * change window function
	 * @param type hamm|hann|rect
	 * @param length window length in ms
	 * @param shift window shift in ms
	 */
	public void setWindowConfig(String type, int length, int shift) {
		
		this.windowLength = length;
		this.windowShift =  shift;
		this.windowType = type;
	}

	/**
	 * helper function (necessary because of memory usage optimization issues)
	 * @param i
	 * @param j
	 * @return
	 */
	protected double getViewSpectrogramBufferAt(int i, int j) {
		
		if(bigSpectrogramBuffer != null) {
			return bigSpectrogramBuffer[i+(viewFromSample-bigFromSample)/samplesPerShift][j];
		} else {
			return 0.;
		}
	}
	
	/**
	 * helper function (necessary because of memory usage optimization issues)
	 * @return
	 */
	protected int getViewSpectrogramBufferLength() {
	
		return viewNSamples/samplesPerShift;
	}
	
	/**
	 * sets the data range (or "zoom" as declared in interface Visualizer)
	 * to be shown in the current spectrogram<br>
	 * fits into the given frame-"raster"<br>
	 * automatically calls computeAndRepaintSpectrogram()
	 * @param fromSample start sample
	 * @param nSamples how many samples to be shown on the image
	 */
	public void setDataRange(int fromSample, int nSamples) {
		
		if (fromSample < 0 || nSamples <= 0) {
			System.err.println("setDataRange: parameters have to be greater than zero!");
			return;
		}
		
		if (fromSample + nSamples > audioFileSize) {
			System.err.println("setDataRange: parameters have to be within range!");
			return;
		}
			
		this.viewFromSample = sampleToFrameRaster(fromSample);
		this.viewNSamples =  sampleToFrameRaster(nSamples);
	
		if (this.viewFromSample >= this.bigFromSample && this.viewFromSample + this.viewNSamples <= this.bigFromSample + this.bigNSamples) {
			repaintAndUpdateImage();
		} else {
			this.bigFromSample = this.viewFromSample;
			this.bigNSamples = this.viewNSamples;

			computeAndRepaintSpectrogram();
		}
	}
		
	
///////////////////////////////////////////////////////////////////////////////
///			methods for computing and visualizing spectrogram data			///
///////////////////////////////////////////////////////////////////////////////
	
	/**
	 * call this function if you want only want to update the image, without
	 * recomputation of the spectrogram data<br>
	 * call this funtcion after setting the colored-flag, logScale-flag, contrast, brightnes, etc.
	 */
	public void repaintAndUpdateImage() {
		
		createSpectrogramImage();
		repaint();
	}

	/**
	 * (re)computes the spectrogram data and stores it into the internal spectrogram buffer<br>
	 * call this function after changing window function, etc.<br>
	 * automatically repaints the spectrogram image
	 */
	public void computeAndRepaintSpectrogram() {
		
		new Runnable() {
			@Override
			public void run() {
				
					BufferedAudioSourceReader basr = bas.getReader(viewFromSample, viewNSamples);	
					if(basr == null) {
						System.err.println("This should not happen :-)");
						return;
					}
					
					bigFromSample = viewFromSample;
					bigNSamples = viewNSamples;
					
					maxFrequency = basr.getSampleRate()/2;
					try {
						window = Window.create(basr, windowType + "," + windowLength + "," + windowShift);
					} catch (MalformedParameterStringException e) {
						e.printStackTrace();
					}
					System.err.println(window);
					FrameSource spec = new FFT(window);
					System.err.println(spec);
					frameSize = spec.getFrameSize();
					
					samplesPerShift = ((basr.getSampleRate() * windowShift) / 1000);
					
					int size = (int) (viewNSamples / samplesPerShift);
					bigSpectrogramBuffer = new double[size][frameSize];

					
					int i = 0;
					try {
						while ((i < bigSpectrogramBuffer.length) && spec.read(bigSpectrogramBuffer[i])) {
							i++;
						}
					} catch (IOException e) {
						e.printStackTrace();
					}
					
					repaintAndUpdateImage();		
			}	
		}.run();
	}
	
	/**
	 * overwrites the current image with data from spectrogram buffer
	 */
	protected void createSpectrogramImage() {
		
		new Runnable() {

			@Override
			public void run() {
						
				for (int x = 0; x < image.getWidth(); x++) {
					
					double ratio = (double)getViewSpectrogramBufferLength() / (double)image.getWidth();
					
					int posXfrom = (int)((double)x * ratio);
					int posXto = (int)((double)(x+1) * ratio);
					
					for (int y = 0; y < image.getHeight(); y++) {
						
						int posY = (int) (y * frameSize / (double) image.getHeight());
						
						double bufferValue = 0.;
						if (image.getWidth() < getViewSpectrogramBufferLength() && useAverage()) {
							for(int k = posXfrom; k < posXto; k++) {
								bufferValue += getViewSpectrogramBufferAt(k, posY);
							}
							bufferValue = bufferValue / (double)(posXto - posXfrom);
						} else {
							bufferValue = getViewSpectrogramBufferAt(posXfrom, posY);
						}
						
						if(getLog()) {
							bufferValue = Math.log(1E-6 + bufferValue);
						}
					
						double value = (bufferValue - min) / ( (max - min));
					
						if (value > 1.) {
							value = 1.;
						}
						
						// scale to gray or color space
						int rgb;
						if (colored) {		
							rgb = Color.HSBtoRGB((float)(1.-value), (float)value, brightness);		
						} 
						else {
							rgb = Color.HSBtoRGB(0.f, 0.f, 1.f-(float)value*brightness);	
						}
						
						image.setRGB(x, image.getHeight() - y - 1, rgb);
					}
				}
				
			}
			
		}.run();
	}
	
	@Override
	public void paint(Graphics g) {
	
		g2d = (Graphics2D) g;
		
		g2d.drawImage(image, 0, 0, this);
		paintOverlay();
	}
	
	/**
	 * paints additional information, markers, etc. on the screen, 
	 * but not in the internal spectrogram image (overlay!)
	 */
	protected void paintOverlay() {
		
		if (getShowMarkers()) {
			
			double start = 0.;
			double end = 0.;
			try {
				start = recomputeMarkerRatio(startMarkerRatio);
				end = recomputeMarkerRatio(endMarkerRatio);
			} catch (Exception e) {
			}
			
			g2d.setColor(markerColor);
			int width = Math.abs((int)((start - end) * (double)image.getWidth()));
			if(startMarkerRatio <= endMarkerRatio) {
				g2d.fillRect((int)(start * image.getWidth()), 0, width, image.getHeight());
			} else {
				g2d.fillRect((int)(end * image.getWidth()), 0, width, image.getHeight());
			}
			g2d.setColor(markerBorderColor);
			g2d.drawLine((int)(start * image.getWidth()), 0, (int)(start * image.getWidth()), image.getHeight());
			g2d.drawLine((int)(end * image.getWidth()), 0, (int)(end * image.getWidth()), image.getHeight());
		}
		
		g2d.setColor(textColor);
		g2d.drawString("Spectrogram", image.getWidth() - 85, 20);
		
		String str = (window.getShift() * (viewNSamples / samplesPerShift)) + " ms";
		g2d.drawString(str, image.getWidth() - str.length()*8 - 10, image.getHeight() - 10);
		
		if (mouseInside) {
			g2d.setColor(scaleColor);
			g2d.drawLine(10, 0, 10, image.getHeight());
			g2d.drawLine(8, image.getHeight()-1, 12, image.getHeight()-1);
			
			double ratio = (double)image.getHeight()/(double)maxFrequency;
			for (int i = 0; i < maxFrequency; i += 1000) {
				int y = image.getHeight() - (int)(ratio * i) - 1;
				g2d.drawLine(8, y, 12, y);
			}
			
			g2d.drawLine(image.getWidth() - 1, mouseY - 1, 8, mouseY - 1);
			ratio = 1. - (double)mouseY/(double)image.getHeight();
			g2d.drawString((int)(ratio * maxFrequency) + " Hz", 13, mouseY - 1);
		}

	}
	
///////////////////////////////////////////////////////////////////////////////
///						internal helper functions							///
///////////////////////////////////////////////////////////////////////////////
	
	/**
	 * converts the given sample number to fit into the frame raster given by
	 * window shift length
	 * @param sample number to be converted
	 * @return converted sample number
	 */
	protected int sampleToFrameRaster(int sample) {
		
		return (sample / samplesPerShift) * samplesPerShift;
	}
	
	/**
	 * internal helper function, only used for paintOverlay()
	 */
	protected double recomputeMarkerRatio(double globalratio) throws Exception {
		
		double curLocalSample = (globalratio * (double)audioFileSize);
		
		if(curLocalSample >= viewFromSample && curLocalSample <= viewFromSample + viewNSamples) {
			return (curLocalSample - viewFromSample) / (double) viewNSamples;
		} else {
			throw new Exception("recomputeMarkerRatio: out of range");
		}
	}
	
	
	/**
	 * opens a JFileChooser Dialog to write the shown image to a .png-file<br>
	 * can be easily modified to work with different image-formats...<br>
	 * really very easy code :-)
	 */
	public void writePNGImageToFile() {
		
		final JFileChooser fc = new JFileChooser();
		fc.setFileFilter(new FileNameExtensionFilter("png", "png"));
		System.out.println("choose a .png file!!!");
		int retVal = fc.showSaveDialog(this);
		if (retVal == JFileChooser.APPROVE_OPTION) {
			File outputfile = fc.getSelectedFile();
		    try {
				ImageIO.write(image, "png", outputfile);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	
///////////////////////////////////////////////////////////////////////////////
///					implementation of internal mouse listeners				///
///		provides a default functionality to control VisualSynchronizer		///
///////////////////////////////////////////////////////////////////////////////
	
	@Override
	public void mouseClicked(MouseEvent e) {
		
		int button = e.getButton(); 	
		if (button == 2) {
			System.out.println("Mouse 2: toggling colored-flag");
			setColored(!getColored());
			if (getColored()) {
				setColorScheme(new Color(0, 0, 0), new Color(100, 100, 100, 50), new Color(0, 0, 0), new Color(100, 100, 100));
			} else {
				setColorScheme(new Color(255, 255, 255), new Color(255, 255, 0, 50), new Color(0, 0, 0), new Color(255, 0, 0));
			}
			repaintAndUpdateImage();
		}
		else if (button == 3) {
			System.out.println("Mouse 3: showing complete audio file");
			setDataRange(0, audioFileSize);
		}
		else if (button == 1) {
			System.out.println("Mouse 1: communicating with known frame diagrams");
			double tmp = (double) e.getX() / (double) getWidth();
			int startSample = (int)(viewFromSample + (double)viewNSamples * tmp);
			startSample = sampleToFrameRaster(startSample);
			
			for (FrameListener listener : listenerList) {
				try {
					listener.show(bas, startSample);
				} catch (Exception e1) {
					e1.printStackTrace();
				}
			}
		}
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		
		mouseInside = true;
		repaint();
	}

	@Override
	public void mouseExited(MouseEvent e) {
		
		mouseInside = false;	
		repaint();
	}

	@Override
	public void mousePressed(MouseEvent e) {

		int button = e.getButton(); 
		if(button == 2) {
			System.out.println("Mouse 2 pressed: drag to zoom");
			markerDragging = true;
			startMarkerRatio = (double) e.getX() / (double) getWidth();
			startMarkerRatio = (viewFromSample + (double)viewNSamples * startMarkerRatio) / audioFileSize;
			endMarkerRatio = startMarkerRatio;
			setShowMarkers(true);
 			setCursor(new Cursor(Cursor.W_RESIZE_CURSOR));
			repaint();
		} 
		
	}

	@Override
	public void mouseReleased(MouseEvent e) {
		
		int button = e.getButton(); 
		if (button == 2) {
			markerDragging = false;
			
			int start = (int)(startMarkerRatio * audioFileSize);
			int end = (int)(endMarkerRatio * audioFileSize);

			if (Math.abs(start - end) <= samplesPerShift) {
				// do nothing
			} else if (endMarkerRatio > startMarkerRatio) {
				setDataRange(start, end - start);
			} else {
				setDataRange(end, start - end);
			}
			setShowMarkers(false);
			setCursor(new Cursor(Cursor.DEFAULT_CURSOR));
		}
	}

	@Override
	public void mouseDragged(MouseEvent e) {
		
		if (markerDragging) {
			endMarkerRatio = (double) e.getX() / (double) getWidth();
			endMarkerRatio = (viewFromSample + (double)viewNSamples * endMarkerRatio) / audioFileSize;
			setShowMarkers(true);
			repaint();
		}
	}

	@Override
	public void mouseMoved(MouseEvent e) {
		
		mouseX = e.getX();
		mouseY = e.getY();
		repaint();	
		
	}

	@Override
	public void mouseWheelMoved(MouseWheelEvent e) {
		
		int rotated =  e.getWheelRotation();
		if (rotated != 0) {
			
			setContrast(getContrast() + (double)rotated);
			repaintAndUpdateImage();
		}
	}

	
/*
 * implementation of ComponentListener
 */
	
	@Override
	public void componentHidden(ComponentEvent e) {}
	
	@Override
	public void componentShown(ComponentEvent e) {}

	@Override
	public void componentMoved(ComponentEvent e) {}

	@Override
	public void componentResized(ComponentEvent e) {
		
		if (image.getHeight() != getHeight() || image.getWidth() != getWidth()) {
			
			if (getHeight() <= 0 || getWidth() <= 0) {
				// do nothing
			} else {
				setImageDimension(getWidth(), getHeight());
				repaintAndUpdateImage();
			}
		}		
	}

/*
 * implementation of interface 'FrameChanger'
 * provides functionality for communication with "per frame"-components, e.g. spectrum, mfcc, ...
 */
	
	@Override
	public void addFrameListener(FrameListener listener) { 
		
		listenerList.add(listener);
		try {
			listener.show(bas, viewFromSample);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

/*
 * implementation of interface 'Visualizer'
 */
	
	@Override
	public int getCurrentEndSample() {
		return viewFromSample + viewNSamples;
	}
	
	@Override
	public int getCurrentStartSample() {
		return viewFromSample;
	}

	@Override
	public int getCurrentSelectionEndSample() {
		
		return (int) (endMarkerRatio * audioFileSize);
	}

	@Override
	public int getCurrentSelectionStartSample() {
	
		return (int) (startMarkerRatio * audioFileSize);
	}

	@Override
	public int getMillisecondsAsSampleNumber(double milliseconds) {
		
		// not needed -> delete this function from interface Visualizer
		return -42;
	}

	@Override
	public double getSampleNumberAsMilliseconds(int sampleIndex) {
		
		// not needed -> delete this function from interface Visualizer
		return -42;
	}

	@Override
	public void resetZoom() {
		
		setDataRange(0, audioFileSize);	
	}

	@Override
	public void select(int startIndex, int endIndex) {
		
		setMarkerRange((double)startIndex / (double)audioFileSize, (double)endIndex / (double)audioFileSize);
		setShowMarkers(true);
		repaint();
	}

	@Override
	public void select(double start, double end) {
		
		setMarkerRange((start * (double)bas.getSampleRate() / 1000.) / (double)audioFileSize, (end * (double)bas.getSampleRate() / 1000.) / (double)audioFileSize);
		setShowMarkers(true);
		repaint();
	}

	@Override
	public void unselect() {
		
		setShowMarkers(false);
		repaint();
	}

	@Override 
	public void zoom(int startIndex, int endIndex) {
		
		setDataRange(startIndex, endIndex - startIndex);
	}

	@Override
	public void zoom(double start, double end) {
		
		setDataRange((int)start * bas.getSampleRate() / 1000, (int)end * bas.getSampleRate() / 1000);
	}
}
