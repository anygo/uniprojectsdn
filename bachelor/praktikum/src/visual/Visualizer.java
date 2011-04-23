package visual;

import java.awt.Dimension;


public interface Visualizer {
	
	/**
	 * Sets the current zoom area that is being shown on the screen to [startIndex, endIndex-1].
	 * @param startIndex the first index (based on the samples array) that is shown within the current area
	 * @param endIndex the first index (based on the samples array) behind the currently shown area
	 */
	public void zoom(int startIndex, int endIndex);
	
	/**
	 * Sets the current zoom area to [start, end]
	 * @param start area begin in milliseconds
	 * @param end area end in milliseconds
	 */
	public void zoom(double start, double end);
	
	/**
	 * Undo all zooming, usually equal to zoom(0, number of all samples)
	 */
	public void resetZoom();
	
	/**
	 * Sets the currently selected samples to [startIndex, endIndex - 1].
	 * @param startIndex the first index (based on the samples array) that is selected
	 * @param endIndex the first index (based on the samples array) behind the selection
	 */
	public void select(int startIndex, int endIndex);
	
	/**
	 * Sets the currently selected samples to [start, end]
	 * @param start selection begin in milliseconds
	 * @param end selection end in milliseconds
	 */
	public void select(double start, double end);
	
	
	/**
	 * Remove the current selection, if there is any.
	 */
	public void unselect();
	
	// set component size: use setPreferredSize(new Dimension(int width, int height))
	// and setSize(new Dimension(int width, int height)) from JComponent
	public void setPreferredSize(Dimension dimension);
	public void setSize(Dimension dimension);
	
	/**
	 * Gets the current zoom area start.
	 * @return the current start index (based on the samples array) of the current zoom area
	 */
	public int getCurrentStartSample();
	
	/**
	 * Gets the current zoom area end.
	 * @return the current end index (based on the samples array) of the current zoom area
	 */
	public int getCurrentEndSample();
	
	/**
	 * Gets the current selection area start.
	 * @return the start index of the current selection, based on the samples array.
	 */
	public int getCurrentSelectionStartSample();
	
	/**
	 * Gets the current selection area end.
	 * @return the end index of the current selection, based on the samples array.
	 */
	public int getCurrentSelectionEndSample();
	
	/**
	 * Changes the sample index into the corresponding milliseconds value.
	 * @param sampleIndex the sample index that should be converted, based on the samples array.
	 * @return the milliseconds
	 */
	public double getSampleNumberAsMilliseconds(int sampleIndex);
	
	/**
	 * Converts milliseconds to a sample index.
	 * @param milliseconds the milliseconds that should be converted
	 * @return the corresponding sample index, based on the samples array.
	 */
	public int getMillisecondsAsSampleNumber(double milliseconds);
	
}