package astar;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;


/**
 * Klasse eines n-Schiebepuzzles
 * 
 * @author zenker
 *
 */
public class NPuzzle implements Cloneable{

	public static final int IMG_SIZE = 250;
	
	private int[][] puzzle;
	
	/**
	 * Das leere Feld
	 */
	public final static int EMPTY = 0;
	
	public final static int MOVE_LEFT = 0;
	public final static int MOVE_RIGHT = 1;
	public final static int MOVE_UP = 2;
	public final static int MOVE_DOWN = 3;
	
	/**
	 * @param n gibt die Groesse des Schiebepuzzles an 
	 */
	public NPuzzle(int n){
		puzzle = new int[n][n];
		
		for(int a = 0; a < n; a++){
			for(int b = 0; b < n; b++){
				puzzle[b][a] = a + (n * b) + 1;
			}
		}
		puzzle[n-1][n-1] = EMPTY;
	}
	
	/**
	 * Ueberprueft, ob ein Zug durchgefuehrt werden kann
	 * @param move Richtung (MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN) 
	 * @return
	 */
	public boolean isMovePossible(int move){
		
		//Position von leerem Feld bestimmen
		int y = 0, x = 0;
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				if(puzzle[b][a] == 0) {
					y = b;
					x = a;
				}
			}
		}
		
		switch(move){
			case MOVE_LEFT:
				if(x > 0)
					return true;
				break;
			case MOVE_RIGHT:
				if(x < puzzle.length - 1)
					return true;
				break;
			case MOVE_UP:
				if(y > 0)
					return true;
				break;
			case MOVE_DOWN:
				if(y < puzzle.length - 1)
					return true;
				break;
			default:
				return false;
		}
		return false;
	}
	
	/**
	 * Verschiebt das freie Feld in die angegebene Richtung
	 * @param move Richtung (MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN) 
	 * @return ob der Zug durchgefuehrt werden konnte
	 */
	public boolean move(int move){
		
		//Position von leerem Feld bestimmen
		int y = 0, x = 0;
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				if(puzzle[b][a] == 0) {
					y = b;
					x = a;
				}
			}
		}
		
		switch(move){
			case MOVE_LEFT:
				if(x > 0){
					puzzle[y][x] = puzzle[y][x - 1];
					puzzle[y][x - 1] = EMPTY;
				}
				return true;
			case MOVE_RIGHT:
				if(x < puzzle.length - 1){
					puzzle[y][x] = puzzle[y][x + 1];
					puzzle[y][x + 1] = EMPTY;
				}
				return true;
			case MOVE_UP:
				if(y > 0){
					puzzle[y][x] = puzzle[y - 1][x];
					puzzle[y - 1][x] = EMPTY;
				}
				return true;
			case MOVE_DOWN:
				if(y < puzzle.length - 1){
					puzzle[y][x] = puzzle[y + 1][x];
					puzzle[y + 1][x] = EMPTY;
				}
				return true;
			default:
				return false;
		}
	}
	
	
	public NPuzzle[] getSuccessors(){
		
		ArrayList<NPuzzle> result = new ArrayList<NPuzzle>();
		
		// i iteriert ueber MOVE_UP, MOVE_DOWN, ...
		for(int i = 0; i <= 3; i++){
			if(isMovePossible(i)){
				NPuzzle p2 = this.clone();
				p2.move(i);
				result.add(p2);
			}
		}
		
		NPuzzle[] r = new NPuzzle[result.size()];
		result.toArray(r);
		return r;
	}

	public BufferedImage getImg()
	{	
		BufferedImage bi = new BufferedImage(puzzle.length, puzzle.length, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bi.createGraphics();
		
		for (int i = 0; i < puzzle.length; ++i)
		{
			for (int j = 0; j < puzzle.length; ++j)
			{
				if (puzzle[i][j] == 0)
				{
					g.setColor(Color.BLACK);
				}
				else
				{	
					float val = ((float)(puzzle[i][j])/(float)(puzzle.length*puzzle.length));
					g.setColor(new Color(Color.HSBtoRGB(val, .8f, .9f)));
					
				}
				g.drawLine(j, i, j, i);
			}
		}
		g.dispose();
		
		BufferedImage resizedImage = new BufferedImage(IMG_SIZE, IMG_SIZE, bi.getType());
		Graphics2D g2 = resizedImage.createGraphics();
		g2.drawImage(bi, 0, 0, IMG_SIZE, IMG_SIZE, null);
		g2.dispose();
		
		return resizedImage;
	}
	
	public String toString(){

		StringBuffer sb = new StringBuffer();
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				sb.append(puzzle[a][b] + "\t");
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	/**
	 * ueberprueft, ob das Puzzle sich im geloesten Zustand befindet 
	 * @return
	 */
	public boolean isSolved(){
		if(puzzle[puzzle.length-1][puzzle.length-1] != EMPTY)
			return false;
		
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				if(a == puzzle.length - 1 && b == puzzle.length - 1){
					return true;
				}
				else if(puzzle[b][a] != a + (puzzle.length * b) + 1){
					return false;
				}
			}
		}
		//Sollte nie erreicht werden:
		return false;
	}
	
	/**
	 * Das Puzzle durcheinander bringen
	 * @param movesCount Anzahl der Verschiebeoperationen
	 */
	public void shuffle(int movesCount){
		
		if(movesCount < 1)
			return;
		
		int lastmove = 0;
		Random zufall = new Random();
		for(int i = 0; i < movesCount; i++){
			int move = zufall.nextInt(4);
			
			//Nur moegliche Verschiebeoperationen ausführen
			while(!isMovePossible(move) || lastmove + move == 1 || lastmove + move == 5)
				move = (move + 1) % 4;
			
			
			
			lastmove = move;
			move(move);
			System.out.println(moveToString(move));
		}
	}
	
	public String moveToString(int move){
		switch(move){
			case MOVE_UP:
				return "up";
			case MOVE_DOWN:
				return "down";
			case MOVE_LEFT:
				return "left";
			case MOVE_RIGHT:
				return "right";
			default:
				return "nooooooo!";
		}
	}
	
	/**
	 * Gibt die Matrix zurueck, welche den aktuellen Zustand des Puzzels enthaelt.
	 * @return
	 */
	public int[][] getPuzzle(){
		return puzzle;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#clone()
	 */
	protected NPuzzle clone(){
		NPuzzle c = new NPuzzle(puzzle.length);
		
		//statt c.puzzle = (int[][]) puzzle.clone();
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				c.puzzle[b][a] = puzzle[b][a];
			}
		}
		return c;
	}
	
	@Override
	public int hashCode(){
		
		StringBuffer sb = new StringBuffer();
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				if(puzzle[a][b] < 10){
					sb.append("0");
				}
				sb.append(puzzle[a][b]);
			}
		}
		
		return sb.toString().hashCode();
	}
	

	@Override
	public boolean equals(Object other) {
		if(other == this) {
		      return true;
		    }
	    if(!(other instanceof NPuzzle)) {
	      return false;
	    }
	   
	    NPuzzle o = (NPuzzle) other;
	    
	    if(this.puzzle.length != o.puzzle.length){
	    	return false;
	    }
	    
		for(int a = 0; a < puzzle.length; a++){
			for(int b = 0; b < puzzle.length; b++){
				if(puzzle[b][a] != o.puzzle[b][a]){
					return false;
				}
			}
		}
		return true;
	}
	
	
    /**
     * Demonstration der Faehigkeiten von NPuzzle
     * @param args
     */
    public static void main(String[] args){
		
    	NPuzzle p = new NPuzzle(3); // 3x3 Puzzle erzeugen
    	NPuzzle p2 = null;
		p2 = (NPuzzle) p.clone();
    	System.out.println(p.move(MOVE_LEFT));
    	System.out.println(p.move(MOVE_LEFT));
    	System.out.println(p.move(MOVE_UP));
    	System.out.println(p.move(MOVE_RIGHT));
		System.out.println(p.toString());
		NPuzzle[] pp = p.getSuccessors(); 
		for(NPuzzle ppp : pp){
			System.out.println(ppp);
		}
    	p.shuffle(10);
    	System.out.println("--------------------");
    	System.out.println(p.toString());
    	System.out.println(p.hashCode());
    	System.out.println("h1: " + Heuristics.h1(p));
//    	System.out.print(p.isMovePossible(MOVE_UP));
    	System.out.println("--------------------");
    	System.out.println("--------------------");
    	System.out.println(p2);
    	System.out.println("--------------------");
		NPuzzle[] s = p2.getSuccessors();
		System.out.println(s[0]);
		System.out.println("--------------------");
		System.out.println(s[1]);
	}
}
