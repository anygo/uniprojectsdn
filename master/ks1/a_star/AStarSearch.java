package astar;

import java.awt.Dimension;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.PriorityQueue;

import javax.swing.JFrame;


public class AStarSearch {

	public static void main(String[] args) throws Exception {
		NPuzzle p = new NPuzzle(6);
		p.shuffle(20);
		//	    	System.out.println(p.move(NPuzzle.MOVE_LEFT));
		//	    	System.out.println(p.move(NPuzzle.MOVE_LEFT));
		//	    	System.out.println(p.move(NPuzzle.MOVE_UP));
		//	    	System.out.println(p.move(NPuzzle.MOVE_RIGHT));

		System.out.println("Zu loesendes Puzzle");
		System.out.println(p);
		System.out.println("==============");
		
		// create frame
		JFrame f = new JFrame();
		f.setVisible(true);
		
		// create panel with selected file
		ImagePanel panelStart = new ImagePanel(p.getImg());
		panelStart.setPreferredSize(new Dimension(NPuzzle.IMG_SIZE, NPuzzle.IMG_SIZE));
		
		// add panel to pane
		f.getContentPane().removeAll();
		f.getContentPane().add(panelStart);
		f.pack();

		NPuzzle[] solution = graphSearch(p);

		if(solution == null) return;

		System.out.println("Ziel gefunden!");
		
		// graphische Ausgabe
		for(int i = 0; i < solution.length; i++)
		{
			System.out.println(solution[i]);
			System.out.println("============");
			
			// create panel with selected file
			ImagePanel panel = new ImagePanel(solution[i].getImg());
			panel.setPreferredSize(new Dimension(NPuzzle.IMG_SIZE, NPuzzle.IMG_SIZE));
			
			// add panel to pane
			f.getContentPane().removeAll();
			f.getContentPane().add(panel);
			f.pack();
			
			Thread.sleep(200);
		}
		
		Thread.sleep(5000);
		System.exit(0);
	}
	  
	public static NPuzzle[] graphSearch(NPuzzle start)
	{
		HashSet<NPuzzle> closed = new HashSet<NPuzzle>();
		PriorityQueue<OpenListEintrag> fringe = new PriorityQueue<OpenListEintrag>();
		
		fringe.add(new OpenListEintrag(start, null, 0));
		
		OpenListEintrag node;
		while ((node = fringe.poll()) != null)
		{
			if (node.destination.isSolved()) 
			{
				int pathLength = 0;
				ArrayList<NPuzzle> al = new ArrayList<NPuzzle>();
				while (node.precessor != null)
				{
					al.add(node.destination);
					node = node.precessor;
					pathLength++;
				}
				System.out.println("Pfadlaenge: " + pathLength);
				NPuzzle[] ret = new NPuzzle[pathLength];
				Collections.reverse(al);
				al.toArray(ret);
				return ret;
			}
			
			if (!closed.contains(node.destination))
			{
				closed.add(node.destination);
				
				// EXPAND
				NPuzzle[] successors = node.destination.getSuccessors();
				for (int i = 0; i < successors.length; ++i)
				{
					OpenListEintrag newNode = new OpenListEintrag(successors[i], node, node.g + Heuristics.h1(successors[i]));
					fringe.add(newNode);
				}
			}
		}
		
		// FAILURE
		return null;
	}
}