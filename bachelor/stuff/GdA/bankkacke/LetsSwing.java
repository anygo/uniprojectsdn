import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class LetsSwing 
{
	public static void main(String[] args)
	{
		JFrame frame = new JFrame("Let's Swing");
		frame.addWindowListener(new WindowAdapter() 
        {	
			public void windowClosing(WindowEvent e) 
			{
				System.exit(0);
			}
		});

		JPanel panel = new JPanel(new BorderLayout());
		frame.getContentPane().add(panel);

		final String[] farbListe = {"Black", "Blue", "Cyan", "Green", "Magenta",
									"Orange", "Red", "White", "Yellow"};

		final JButton resetButton = new JButton("Reset");
		final JList list = new JList(farbListe);
		
		list.addListSelectionListener(new ListSelectionListener()
		{
			public void valueChanged(ListSelectionEvent e)
			{
				String farbe = farbListe[list.getSelectedIndex()];
				if (farbe.equals("Black")) resetButton.setForeground(Color.black);
				if (farbe.equals("Blue")) resetButton.setForeground(Color.blue);
				if (farbe.equals("Cyan")) resetButton.setForeground(Color.cyan);
				if (farbe.equals("Green")) resetButton.setForeground(Color.green);
				if (farbe.equals("Magenta")) resetButton.setForeground(Color.magenta);
				if (farbe.equals("Orange")) resetButton.setForeground(Color.orange);
				if (farbe.equals("Red")) resetButton.setForeground(Color.red);
				if (farbe.equals("White")) resetButton.setForeground(Color.white);
				if (farbe.equals("Yellow")) resetButton.setForeground(Color.yellow);
			}
		});

		resetButton.addActionListener(new ActionListener() 
		{
			public void actionPerformed(ActionEvent e){
				{
					resetButton.setForeground(Color.black);
					list.setSelectedIndex(0);
				}
			}
		});
		
		panel.add(list, BorderLayout.CENTER);	
		panel.add(resetButton, BorderLayout.SOUTH);
		frame.pack();
		frame.setSize(160, 202);
		frame.setVisible(true);
	}
}
