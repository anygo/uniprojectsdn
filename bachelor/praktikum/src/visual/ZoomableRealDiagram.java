package visual;

import java.awt.BasicStroke;
import java.awt.Cursor;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

/**
 * a class exntending the VisualRealDiagram with additional functionality of zooming. It reacts to mouse-events and starts the zoom process.
 * @author Matthias Ring, Markus Goetze
 *
 */
public abstract class ZoomableRealDiagram extends VisualRealDiagram implements MouseMotionListener, MouseListener {
	
	private static final long serialVersionUID = 1L;
	
	/**
	 * position of mouse once button is pressed, corresponds to zoom begin position
	 */
	protected int startMouseDragX = -1;
	
	/**
	 * position of mouse once button is released, corresponds to zoom end position
	 */
	protected int endMouseDragX = -1;
	
	/**
	 * determines if mouse button was released, major use to differentiate between mouse click and drag
	 */
	protected boolean mouseDragReleased = false;
	
	/**
	 * determines if mouse button was pressed, major use to differentiate between mouse click and drag
	 */
	protected boolean mouseDragPressed = false;
	
	/**
	 * current mouse position for big cross hair
	 */
	protected int mouseX = -1;

	
	/**
	 * 
	 * @param title Title to be shown
	 * @param log determines whether to logarithmize
	 * @param maxFrequency maximum frequency
	 */
	protected ZoomableRealDiagram(String title, boolean log, double maxFrequency, String xAxisLabel, boolean spectrumDiagram) {
		super(title, log, maxFrequency, xAxisLabel, spectrumDiagram);
		addMouseMotionListener(this);
		addMouseListener(this);
	}
			
	@Override
	public void mouseEntered(MouseEvent e) {
	}

	/**
	 * if the button pressed is the right mouse button, zoom out and repaint the image
	 */
	@Override
	public void mouseClicked(MouseEvent e) {
		if (e.getButton() == MouseEvent.BUTTON3) {
			mouseDragReleased = false;
			startMouseDragX = -1;
			endMouseDragX = -1;
			zoomOut();
			repaint();
		}
	}

	/**
	 * sets a flag that mouse dragging is in progress
	 */
	@Override
	public void mouseDragged(MouseEvent e) {
		mouseX = e.getX();
		mouseDragPressed = true;

		repaint();
	}

	/**
	 * update current mouse position and repaint to get new cross hair
	 */
	@Override
	public void mouseMoved(MouseEvent e) {
		mouseX = e.getX();
		repaint();
	}

	/**
	 * if mouse position leaves window, reset flags, reset cursor type and repaint to remove cross hair
	 */
	@Override
	public void mouseExited(MouseEvent e) {
		mouseX = -1;
		startMouseDragX = -1;
		mouseDragPressed = false;
		setCursor(new Cursor(Cursor.CROSSHAIR_CURSOR));
		repaint();
	}

	@Override
	public void mousePressed(MouseEvent e) {
		startMouseDragX = e.getX();
		setCursor(new Cursor(Cursor.HAND_CURSOR));
	}

	/**
	 * save end position of mouse drag, reset mouseDragPressed flag to stop drawing yellow rectangle and zoom in
	 */
	@Override
	public void mouseReleased(MouseEvent e) {
		endMouseDragX = e.getX();
		mouseX = e.getX();
		mouseDragReleased = true;
		mouseDragPressed = false;
		
		zoomIn();
		repaint();
		
		setCursor(new Cursor(Cursor.CROSSHAIR_CURSOR));
	}

	/**
	 * reset flags after zooming
	 */
	protected void enableNewZoom() {
		mouseDragReleased = false;
		startMouseDragX = -1;
		endMouseDragX = -1;
	}
	
	/**
	 * Draw yellow zooming rectangle and cross hair of mouse position on x-axis and the corresponding value on y-axis
	 * @param bg graphics on which to be drawn
	 * @param w width of window considering border indent
	 * @param h height of window
	 */
	protected void mouseMoveDrawingEvents(Graphics2D bg, int w, int h) {
		double xValue;
		int dataPointer;
		int pixelPointer;

		// draw frequency line and big crosshair if mouse inside window
		if (mouseX != -1 && mouseX >= indentBorder && mouseX < w + indentBorder) {
			// zomming window
			if (mouseDragPressed) {
				drawRectangle(bg, startMouseDragX, 0, mouseX, h);
			}

			// draw big crosshair
			xValue = currentMaxXValue - currentMinXValue;
			xValue = xValue / w;
			xValue = xValue * (double) (mouseX - indentBorder);
			xValue = currentMinXValue + xValue;
			xValue = roundTwoDecimals(xValue);
			pixelPointer = mouseX - indentBorder;
			dataPointer = (int) (((double) mouseX - indentBorder) * valsPerPix);

			drawDashedLine(bg, mouseX, 0, mouseX, h);
			drawDashedLine(bg, 0, (int) pixelBuffer[pixelPointer], getWidth(),
					(int) pixelBuffer[pixelPointer]);
			
			FontMetrics fm = bg.getFontMetrics();
			double value = copyBuffer[dataPointer];
			double diff = (getMax() + getMin()) / 2;
			if (value >= 0) {
				value = value * Math.abs(getMax() - diff) + diff;
			} else {
				value = value *  Math.abs(getMin() - diff) + diff;
			}
			String text = "" + xValue + getxAxisLabel() + ", "	+ roundTwoDecimals(value) + "";
			int valueLength = fm.stringWidth(text);
			bg.drawString(text, getWidth() - valueLength - indentBorder, h - 10);
		}
	}

	
	/**
	 * draws the rectangle displayed when mouse is dragged for zoom functionality
	 * @param bg Graphics on which to be drawn
	 * @param startX	start position of mouse drag on x-axis
	 * @param startY	start position of mouse drag on y-axis
	 * @param endX		end position of mouse drag on x-axis
	 * @param endY		end position of mouse drag on x-axis
	 */
	protected void drawRectangle(Graphics2D bg, int startX, int startY,
			int endX, int endY) {
		bg.setColor(SELECTION_COLOR);

		if (endX < startX) {
			bg.fill3DRect(endX, startY, startX - endX, endY, true);
		} else {
			bg.fill3DRect(startX, startY, endX - startX, endY, true);
		}
	}

	/**
	 * draws a dashed line
	 * @param bg	graphics on which to be drawn
	 * @param x1	start x-position
	 * @param y1	start y-position
	 * @param x2	end x-position
	 * @param y2	end y-position
	 */	
	protected void drawDashedLine(Graphics2D bg, int x1, int y1, int x2, int y2) {
		Stroke stroke_save = bg.getStroke();
		bg.setColor(CROSSHAIR_COLOR);
		BasicStroke stroke = new BasicStroke(1, BasicStroke.CAP_BUTT,
				BasicStroke.JOIN_BEVEL, 1, new float[] { 1 }, 0);
		bg.setStroke(stroke);
		bg.drawLine(x1, y1, x2, y2);
		bg.setStroke(stroke_save);
	}

	/**
	 * Zoom in and display the selected range. Adjust axes labels.
	 */
	protected void zoomIn() {
		int startPointer;
		int endPointer;
		double[] tmp;
		int swp;
		int w = getWidth() - 2 * indentBorder;

		if (startMouseDragX > -1
				&& Math.abs(endMouseDragX - startMouseDragX) > 10
				&& mouseDragReleased) {
			startPointer = (int) (((double) startMouseDragX - indentBorder) * valsPerPix);
			endPointer = (int) (((double) endMouseDragX - indentBorder) * valsPerPix);

			if (endPointer < startPointer) {
				swp = startPointer;
				startPointer = endPointer;
				endPointer = swp;
			}

			// adjust frequency
			double endSave = currentMaxXValue;
			double startSave = currentMinXValue;
			currentMaxXValue = (endSave - startSave)
					* ((double) endPointer / (double) copyBufferSize)
					+ currentMinXValue;
			currentMinXValue = (endSave - startSave)
					* ((double) startPointer / (double) copyBufferSize)
					+ currentMinXValue;

			copyBufferSize = endPointer - startPointer;
			if (copyBufferSize < 10) {
				copyBufferSize = 10; // max zoom = two points
				currentMaxXValue = currentMinXValue;
			}
			valsPerPix = copyBufferSize / (double) w;

			// copy buffer adjust
			tmp = new double[Math.abs(copyBufferSize)];
			System.arraycopy(copyBuffer, startPointer, tmp, 0, copyBufferSize);
			System.arraycopy(tmp, 0, copyBuffer, 0, copyBufferSize);

			// pixel buffer adjust
			createPixelBuffer(w);

			enableNewZoom();
		}
	}
	
	/**
	 * Reset displayed range to original data.
	 */
	protected void zoomOut() {
		System.arraycopy(this.originalBuffer, 0, this.copyBuffer, 0, frameSize);
		copyBufferSize = frameSize;
		currentMinXValue = 0;
		currentMaxXValue = getMaxXValue();
	}
	
	/**
	 * Draw the image and additionally draw mouse moving events (zooming rectangle and cross hair).
	 */
	@Override
	public void paint(Graphics bg) {
		super.paint(bg);
		mouseMoveDrawingEvents((Graphics2D)bg, (getWidth() - 2 * indentBorder), getHeight());
	}

}
