package astar;

import java.awt.Graphics;
import java.awt.Image;
import javax.swing.JPanel;

public class ImagePanel extends JPanel
{	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	//image object
	private Image img;
	
	public ImagePanel(Image img)
	{
		this.img = img;		
	}
	
	//override paint method of panel
	public void paint(Graphics g)
	{
		if (img != null)
			g.drawImage(img, 0, 0, this);
	}
	
}