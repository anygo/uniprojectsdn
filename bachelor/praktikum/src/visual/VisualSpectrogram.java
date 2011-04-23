package visual;

import java.awt.Color;
import java.awt.image.BufferedImage;
import javax.swing.JComponent;

/**
 * super Visualizer class that provides common methods and variables
 * 
 * @author sidoneum, sifeluga
 * 
 */

public abstract class VisualSpectrogram extends JComponent {

	private static final long serialVersionUID = 1L;
	
	/**
	 * the image on which the spectrogram data is written
	 */
	protected BufferedImage image = null;
	
	/**
	 * upper limit of frequency range: sampleRate/2
	 */
	protected int maxFrequency = 0;
	
	/**
	 * number of values per frame
	 */
	protected int frameSize = 0;
	
	/**
	 * limits the lower range of values
	 */
	protected double min = 0.;
	
	/**
	 * limits the upper range of values
	 */
	protected double max = 1.;
	
	/**
	 * specifies the intensity of the image brightness
	 */
	protected float brightness = 1.f;
	
	/**
	 * use logarithmic range of values
	 */
	protected boolean log = true;
	
	/**
	 * indicator of a colored spectrogram
	 */
	protected boolean colored;

	
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
	
	
	protected Color scaleColor = new Color(255, 0, 0);
	protected Color textColor = new Color(255, 255, 255);
	
	
///////////////////////////////////////////////////////////////////////////////
///							set- and get-methods							///
///////////////////////////////////////////////////////////////////////////////
	
	
	/**
	 * sets the spectrogram image to be grayscale or of color
	 * @param colored false = grayscale, true = colored
	 */
	public void setColored(boolean colored) {
		
		this.colored = colored;
	}
	
	/**
	 * returns if the spectrogram is colored or not
	 * @return false = grayscale, true = colored
	 */
	public boolean getColored() {
		
		return colored;
	}
	
	
	/**
	 * sets whether the spectrogram uses logarithmic values or not
	 * @param log
	 */
	public void setLog(boolean log) {
		
		this.log = log;
		
		if(log) {
			this.min = Math.log(1E-6);
		} else {
			this.min = 0;
		}
	}
	
	/**
	 * returns if the spectrogram uses logarithmic values or not
	 * @return
	 */
	public boolean getLog() {
		
		return log;
	}
	
	/**
	 * sets the new dimension of the spectrogram image
	 * @param width the new width
	 * @param height the new height
	 */
	protected void setImageDimension(int width, int height) {

		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	}
	
	/**
	 * controls the brightness of the spectrogram image
	 * @param brightness float value between 0.f and 1.f 
	 */
	public void setBrightness(float brightness) {
		if(brightness < 0) {
			this.brightness = 0.f;
		} else if(brightness > 1.f) {
			this.brightness = 1.f;
		} else {
			this.brightness = brightness;
		}	
	}
	
	/**
	 * 
	 * @return current brightness of the spectrogram
	 */
	public float getBrightness() {
		
		return brightness;
	}

	/**
	 * sets the contrast of the spectrogram
	 * @param contrast > 0.
	 */
	public void setContrast(double contrast) {
		
		this.max = contrast;
		if (this.max <= 0) {
			this.max = 0.;
		}
		
		if(getLog()) {
			this.min = Math.log(1E-6);
		} else {
			this.min = 0;
		}
	}
	
	/**
	 * returns contrast of spectrogram
	 * @return current contrast of the spectrogram
	 */
	public double getContrast() {
		
		return max;
	}
}
