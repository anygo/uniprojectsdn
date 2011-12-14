import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;


/**
 * Solver for so-called "Alphametiken" based on recursive depth-first search
 * Popular examples for Alphametiken:
 * "SEND + MORE = MONEY" or "HACKER + CRASH = REBOOT"
 * 
 * @author Dominik Neumann
 * @version 0.1
 *
 */
public class AlphametikSolver {

	private String symbolsA; 	//   A
	private String symbolsB; 	// + B
	private String symbolsC;	// = C
	private HashMap<Character, Integer> symbolAssignments; // holds assignments for symbols
	private long checkCounter; // counts evaluations of check()-primitive
	private long solutionsCounter;
	
	/**
	 * Constructor
	 * @param A first word containing its symbols (chars)
	 * @param B second word containing its symbols (chars)
	 * @param SUM third word containing its symbols (chars)
	 */
	public AlphametikSolver(String A, String B, String SUM) {
		
		this.symbolsA = A;
		this.symbolsB = B;
		this.symbolsC = SUM;
		
		this.symbolAssignments = new HashMap<Character, Integer>(); // char/symbol -> digit
		checkCounter = 0;
		solutionsCounter = 0;
		
		// fill set with characters from all strings (HashMap -> unique keys (unique symbols/chars))
		addCharsToSymbolMap(symbolsA);
		addCharsToSymbolMap(symbolsB);
		addCharsToSymbolMap(symbolsC);	
	}
	
	/**
	 * adds characters of a string to HashMap (symbols -> digits)
	 * @param str String containing chars/symbols, that will be added to the map
	 */
	private void addCharsToSymbolMap(String str) {
		
		for (int i = 0; i < str.length(); ++i) {
			symbolAssignments.put(str.charAt(i), 0);
		}
	}
	
	/**
	 * Convert symbols of str to decimal number given the current symbol assignments
	 * @param str String used for conversion
	 * @return
	 */
	private int convertToNumber(String str) {
		
		int sum = 0;
		for (int i = 0; i < str.length(); ++i) {			
			sum += symbolAssignments.get(str.charAt(str.length()-i-1)) * Math.pow(10, i);
		}
		
		return sum;
	}
	
	/**
	 * Checks whether we found a valid solution
	 * @return true, if current assignment is valid, false otherwise
	 */
	private boolean check() {	
		
		// increment counter
		++checkCounter;
		
		// we already have unique digits, so we do not have to check for that
		return (convertToNumber(symbolsA) + convertToNumber(symbolsB) == convertToNumber(symbolsC));
	}
	
	/**
	 * Recursive DFS search
	 * @param it Iterator pointing to current symbol in HashMap
	 * @param availableDigits Set holding all possible digits (i.e. digits that are not used for any other symbol)
	 * @return true, if a valid solution has been found
	 */
	private void search(Iterator<Character> it, HashSet<Integer> availableDigits) {
		
		// recursion: break condition
		if (!it.hasNext()) {
			
			// check for validity
			if (check()) {

				solutionsCounter++;
				
				// print solution and some additional info
				printAssignments();
				System.out.println();
				printSolution();
				printInfo();
				System.out.println("-----------------------------------------------------------------------------------\n");
			}
			return;
		}
		
		// get current symbol
		Character sym = it.next();
		
		// try all possibilities for current symbol
		for (Integer digit: availableDigits) {
			
			// update digit for current symbol
			symbolAssignments.put(sym, digit);
		
			// create new set of available numbers w/o digit for current symbol sym
			HashSet<Integer> newAvailableDigits = new HashSet<Integer>();
			newAvailableDigits.addAll(availableDigits);
			newAvailableDigits.remove(digit);			
			
			// iterator is not cloneable, so we manually 'clone' it
			Iterator<Character> itCopy = symbolAssignments.keySet().iterator();
			while (itCopy.next() != sym);
			
			// continue recursion
			search(itCopy, newAvailableDigits);			
		}
		
		// no valid solution found
		return;
	}
	
	/**
	 * Initiates the recursive DFS, call this function from outside
	 */
	public void solve() {
		
		// create set with all digits [0..9]
		HashSet<Integer> availableNumbers = new HashSet<Integer>();
		for (int i = 0; i < 10; ++i) {
			availableNumbers.add(i);
		}
		
		// start recursive DFS
		search(symbolAssignments.keySet().iterator(), availableNumbers);
	}
	
	/**
	 * Returns number of valid solutions
	 * @return number of valid solutions
	 */
	public long getNumSolutions() {
		
		return solutionsCounter;
	}
	
	/**
	 * Prints information about number of function evaluations to System.out
	 */
	public void printInfo() {
		
		float fac = 1;
		for (int i = 0; i < symbolAssignments.size(); ++i)
			fac *= (10-i);
		float percentage = (float)checkCounter / fac * 100.f;
		System.out.println("# evaluations of check() so far: " + checkCounter + " (" + percentage + "% of all possible calls)");
	}
	
	/**
	 * Prints current symbol-digit assignments to System.out
	 */
	public void printAssignments() {
		
		System.out.println("Assignments:");
		for (Character c: symbolAssignments.keySet()) {
			System.out.print(c + " := " + symbolAssignments.get(c) + "   ");
		}
		System.out.println();
	}
	
	/**
	 * Prints current valid solution to System.out
	 */
	public void printSolution() {
		
		final int maxLength = Math.max(symbolsA.length(), Math.max(symbolsB.length(), symbolsC.length()));
		for (int i = 0; i < maxLength-symbolsA.length()+1; ++i)
			System.out.print(' ');
		for (Character c: symbolsA.toCharArray()) 
			System.out.print(symbolAssignments.get(c));
		for (int i = 0; i < maxLength-symbolsA.length()+1; ++i)
			System.out.print(' ');
		System.out.print("(" + symbolsA + ")\n+");
		for (int i = 0; i < maxLength-symbolsB.length(); ++i)
			System.out.print(' ');
		for (Character c: symbolsB.toCharArray()) 
			System.out.print(symbolAssignments.get(c));
		for (int i = 0; i < maxLength-symbolsB.length()+1; ++i)
			System.out.print(' ');
		System.out.print("(" + symbolsB + ")\n");
		for (int i = 0; i < (maxLength+1)*2+2; ++i)
			System.out.print('-');
		System.out.print("\n=");
		for (int i = 0; i < maxLength-symbolsC.length(); ++i)
			System.out.print(' ');
		for (Character c: symbolsC.toCharArray()) 
			System.out.print(symbolAssignments.get(c));
		for (int i = 0; i < maxLength-symbolsC.length()+1; ++i)
			System.out.print(' ');
		System.out.print("(" + symbolsC + ")\n\n");
	}
	
	/**
	 * Main-method
	 * @param args
	 */
	public static void main(String[] args) {
		
		// store symbols
		String[] words;
		
		// check input args
		if (args.length != 3) {
			System.err.println("Usage: java AlphametikSolver WORD_A WORD_B WORD_SUM");
			System.err.println("  -> fallback: \"HACKER CRASH REBOOT\"\n");
			
			words = new String[3];
			words[0] = "HACKER";
			words[1] = "CRASH";
			words[2] = "REBOOT";
		} else {
			words = args;
		}
		
		// instantiate solver
		AlphametikSolver lp = new AlphametikSolver(words[0], words[1], words[2]);
		
		System.out.println("Trying to find all solutions... ");
		
		// solve and print solutions, if appropriate
		lp.solve();
			
		if (lp.getNumSolutions() == 0) {
			System.out.println("No solution.");
		} else {
			System.out.println("Total number of solutions: " + lp.getNumSolutions() + ".");
		}
	}
}
