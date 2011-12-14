package astar;

public class Heuristics {

	/**
	 * Heuristik: Anzahl der falsch besetzten Felder
	 * 
	 * @param p n-Puzzle
	 * @return
	 */
	public static int h1(NPuzzle p){
		
		int[][] m = p.getPuzzle();	
		int n = m.length;
		int cnt = 0;
		
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (m[i][j] == NPuzzle.EMPTY)
				{
					if (i != n-1 || j != n-1)
					{
						cnt++;
					}
				}
				else 
				{
					if (m[i][j]-1 != i*n + j)
					{
						cnt++;
					}
				}
			}
		}
		
		return cnt;
	}
	
	/**
	 * Heuristik: Summe der Manhattan-Entfernungen von falsch besetzten Felden zu deren Zielposition
	 * 
	 * @param p n-Puzzle
	 * @return
	 */
	public static int h2(NPuzzle p){

		int[][] m = p.getPuzzle();	
		int n = m.length;
		int totalDist = 0;
		
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				int xPos = j % n;
				int yPos = i / n;
				
				int curItem = m[i][j];
				
				// EMPTY == 0, gehoert aber eigentlich an m[n-1][n-1]
				if (curItem == NPuzzle.EMPTY)
				{
					curItem = n*n;
				}

				int xPosGoal = (curItem-1) % n;
				int yPosGoal = (curItem-1) / n;
				
				totalDist += Math.abs(xPos - xPosGoal) + Math.abs(yPos - yPosGoal);
			}
		}
		
		return totalDist;
	}
}
