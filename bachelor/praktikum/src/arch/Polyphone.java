package arch;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;

import statistics.HMM;
import util.Pair;

/**
 * A polyphone consists of a central phone, its left and right context and the 
 * statistical models linked to it. The polyphone inventory is fixed, the tree
 * structure references to these instances on demand.
 * 
 * @author sikoried
 *
 */
public class Polyphone implements Serializable {
	private static final long serialVersionUID = 1L;
	
	/** syllable boundary */
	public static final String SB = "|";
	
	/** word boundary */
	public static final String WB = "#"; 

	/** left context, including boundaries */
	String[] left;
	
	/** right context, including boundaries */
	String[] right;
	
	/** central phone */
	String phone;
		
	/** polyphone w/ less context, if any */
	Polyphone lessContext = null;
	
	/** polyphone(s) w/ more context, if any */
	Polyphone [] moreContext = new Polyphone [0];
	
	/** internal hash String, requires update if context is changed */
	private String hash;
	
	/** internal hash number, reuires update if context is changed */
	private int hashval;
	
	/** instance counter */
	private static long instances = 0;
	
	/** unique ID */
	public final long INSTANCE_ID = instances++;
	
	/** number of occurrences in the training set */
	public int occurrences = 0;
	
	/** acoustic model associated with this polyphone */
	public HMM hmm;
	
	/**
	 * Create a polyphone with the given context.
	 * @param left
	 * @param phone
	 * @param right
	 */
	public Polyphone(String [] left, String phone, String [] right) {
		this.left = left;
		this.right = right;
		this.phone = phone;
		updateHash();
	}
	
	/**
	 * Create a monophone without context.
	 * @param monophone
	 */
	public Polyphone(String monophone) {
		this.left = new String [0];
		this.right = new String [0];
		this.phone = monophone;
		updateHash();
	}
	
	/**
	 * If the context of the phone (or the phone itself) was changed, you need
	 * to update the hash!
	 */
	void updateHash() {
		StringBuffer sb = new StringBuffer();
		for (String s : left)
			sb.append(s);
		sb.append("/" + phone + "/");
		for (String s : right)
			sb.append(s);
		hash = sb.toString();
		hashval = hash.hashCode();
	}
	
	public boolean isMonophone() {
		return (left.length == 0 && right.length == 0);
	}
	
	public boolean isBiphone() {
		return (left.length == 0 && right.length == 1);
	}
	
	public boolean isTriphone() {
		return (left.length == 1 && right.length == 1);
	}
	
	/**
	 * Reset the occurrence counter of this polyphone and its children
	 */
	public void resetOccurrenceCounter() {
		occurrences = 0;
		for (Polyphone p : moreContext)
			p.resetOccurrenceCounter();
	}
	
	/**
	 * Add a Polyphone to the hierarchy
	 * @param child
	 */
	public void addChild(Polyphone child) {		
		// examine the children
		for (int i = 0; i < moreContext.length; ++i) {
			Polyphone cand = moreContext[i];
		
			// descend in hierarchy?
			if (child.specializes(cand)) {
				cand.addChild(child);
				return;
			}
			
			// one step too far?
			if (child.generalizes(cand)) {
				// insert the child, update the parent relation, add the replaced candidate
				moreContext[i] = child;
				child.lessContext = this;
				child.addChild(cand);
				
				// attention now: there might be more generalizing candidates on this level, check this!
				if (i != moreContext.length - 1) {
					// build up a new moreContext list: add all already checked children
					ArrayList<Polyphone> newContext = new ArrayList<Polyphone>();
					ArrayList<Polyphone> addToChild = new ArrayList<Polyphone>();
					
					for (int j = 0; j <= i; ++j)
						newContext.add(moreContext[j]);
					
					// search for potential candidates
					for (int j = i + 1; j < moreContext.length; ++j) {
						if (child.generalizes(moreContext[j]))
							addToChild.add(moreContext[j]);
						else
							newContext.add(moreContext[j]);
					}
					
					// did the context change at all?
					if (addToChild.size() > 0) {
						moreContext = newContext.toArray(new Polyphone [moreContext.length - addToChild.size()]);
						for (Polyphone p : addToChild)
							child.addChild(p);
					}
				}
				return;
			}
		}
		
		// we're good to insert here!
		Polyphone [] expanded = new Polyphone [moreContext.length + 1];
		System.arraycopy(moreContext, 0, expanded, 0, moreContext.length);
		expanded[expanded.length-1] = child;
		moreContext = expanded;
		child.lessContext = this;
	}
	
	/**
	 * Prune the phoneme hierarchy to remove extra "idle" links. This is handled
	 * through recursion!
	 */
	public void pruneHierarchy() {
		// down at the very bottom -- nothing to do
		if (moreContext.length == 0)
			return;
		
		// DFS by recursion
		for (Polyphone p : moreContext) {
			p.pruneHierarchy();
			
			if (p.moreContext.length == 1) {
				p.moreContext = p.moreContext[0].moreContext;
				for (Polyphone c : moreContext)
					c.lessContext = this;
			}
		}
		
		if (moreContext.length == 1) {
			moreContext = moreContext[0].moreContext;
			for (Polyphone c : moreContext)
				c.lessContext = this;
		}
	}
	
	/**
	 * Prune the phoneme hierarchy by removing polyphones appearing less than
	 * minOcc times (recursive call).
	 * 
	 * @param minOcc minimum number of occurrences
	 */
	public void pruneHierarchyByOccurrence(int minOcc) {
		if (moreContext.length == 0)
			return;
		
		ArrayList<Polyphone> prunedContext = new ArrayList<Polyphone>();
		for (Polyphone p : moreContext) {
			// if we keep p, we need to prune its hierarchy
			if (p.occurrences >= minOcc) {
				p.pruneHierarchyByOccurrence(minOcc);
				prunedContext.add(p);
			}
		}
		
		// if there was any pruning update the moreContext array
		if (prunedContext.size() != moreContext.length)
			moreContext = prunedContext.toArray(new Polyphone [prunedContext.size()]);
	}
	
	/**
	 * Obtain a string representation of the polyphone, e.g. raI/s/@
	 */
	public String toString() {
		return hash; // + " (" + occurrences + ")";
	}
	
	/**
	 * Generate a String representation of the hierarchy using the .dot format
	 * @return .dot graph without header
	 */
	public String hierarchyAsDotFormat(boolean includeHeader) {
		// iterative approach
		ArrayList<Polyphone> agenda = new ArrayList<Polyphone>();
		agenda.add(this);
		
		StringBuffer sb = new StringBuffer();
		
		if (includeHeader) {
			sb.append("digraph Polyphones {\n" +
				"ordering=out;\n" +
				"rankdir=LR;\n" +
				"node [shape=box];\n");
		}
		
		// DFS search
		while (agenda.size() > 0) {
			Polyphone p = agenda.remove(agenda.size() - 1);
			sb.append("node_" + p.INSTANCE_ID + " [label=\"" + p.toString() + "\"];\n");
			if (p.lessContext != null)
				sb.append("node_" + p.INSTANCE_ID + " -> node_" + p.lessContext.INSTANCE_ID + ";\n");
			for (Polyphone child : p.moreContext)
				agenda.add(child);
		}
		
		if (includeHeader)
			sb.append("}\n");
		
		return sb.toString();
	}
	
	/**
	 * Generate a String representation of the hierarchy using ASCII art
	 * @return 
	 */
	public String hierarchyAsString() {
		StringBuffer sb = new StringBuffer();
		ArrayList<Pair<Integer, Polyphone>> agenda = new ArrayList<Pair<Integer, Polyphone>>();
		agenda.add(new Pair<Integer, Polyphone>(0, this));
		final String INDENT = "    ";

		// depth first search
		while (agenda.size() > 0) {
			Pair<Integer, Polyphone> pair = agenda.remove(agenda.size() - 1);

			// do correct indent
			for (int i = 0; i < pair.a; ++i)
				sb.append(INDENT);

			// print current
			sb.append(pair.b.toString());

			// add the children
			for (Polyphone child : pair.b.moreContext)
				agenda.add(new Pair<Integer, Polyphone>(pair.a + 1, child));

			// finish line
			sb.append("\n");
		}
		return sb.toString();
	}
	
	/**
	 * Obtain a hash value using the string representation of the polyphone
	 */
	public int hashCode() {
		return hashval;
	}
	
	public boolean equals(Polyphone p) {
		return p.hash.equals(hash);
	}
	
	public boolean equals(String phoneInContext) {
		return hash.equals(phoneInContext);
	}
	
	public boolean equals(Object o) {
		if (o instanceof Polyphone)
			return equals((Polyphone) o);
		else
			return false;
	}
	
	/**
	 * Check whether or not the polyphone specializes the referenced polyphone.
	 * NB: An equal polyphone is not a specialization! 
	 * 
	 * @param p 
	 * @return
	 */
	public boolean specializes(Polyphone p) {
		if (!p.phone.equals(phone) || p.equals(this))
			return false;
		
		// check if context lengths match
		if (p.left.length > left.length || p.right.length > right.length)
			return false;
		
		// check if right context matches
		for (int i = 0; i < p.right.length; ++i)
			if (!right[i].equals(p.right[i]))
				return false;
		
		// check if left context matches; mind the direction and that p.left.length is shorter than left!
		int offset = left.length - p.left.length;
		for (int i = 0; i < p.left.length; ++i)
			if (!left[offset + i].equals(p.left[i]))
				return false;
		
		// everything matches, p is enclosed in this polyphone!
		return true;
	}
	
	/**
	 * Check whether or not the polyphone generalizes the referenced polyphone
	 * NB: An equal polyphone is not a generalization!
	 * 
	 * @param p
	 * @return
	 */
	public boolean generalizes(Polyphone p) {
		if (!p.phone.equals(phone) || p.equals(this))
			return false;
		
		// check if context lengths match
		if (left.length > p.left.length || right.length > p.right.length)
			return false;
		
		// check if right context matches
		for (int i = 0; i < right.length; ++i)
			if (!right[i].equals(p.right[i]))
				return false;
		
		// check if left context matches; mind the direction and that p.left.length is shorter than left!
		int offset = p.left.length - left.length;
		for (int i = 0; i < left.length; ++i)
			if (!left[i].equals(p.left[offset+i]))
				return false;
		
		// everything matches, p is enclosed in this polyphone!
		return true;
	}
	
	/**
	 * Extract all possible polyphones from the given word transcription.
	 * 
	 * @param trans Transcription of a single word in form of a String 
	 *        array of phonemes. Make sure you have leading and trailing word
	 *        boundaries and syllabe boundaries (if desired).
	 * @return Array of polyphones (not linked)
	 */
	public static Polyphone [] extractPolyphonesFromWordTranscription(String [] trans) {
		LinkedList<Polyphone> phones = new LinkedList<Polyphone>();
		
		// this will be the growing context
		ArrayList<String> left = new ArrayList<String>();
		ArrayList<String> right = new ArrayList<String>();
		
		// For each center phoneme, get all possible polyphones
		for (int i = 1; i < trans.length - 1; ++i) {
			// We're only interested in phonemes!
			if (trans[i].equals(Polyphone.SB) || trans[i].equals(Polyphone.WB))
				continue;
			
			// add the monophone
			phones.add(new Polyphone(trans[i]));
			
			// now generate every possible polyphone, expand right-left-right-...
			// until both contexts are exploited
			int il = i-1;
			int ir = i+1;
			
			while (il >= 0 || ir < trans.length) {
				// expand right
				if (ir < trans.length) {
					right.add(trans[ir]);
					ir++;
					
					phones.add(new Polyphone(
							left.toArray(new String [left.size()]), 
							trans[i], 
							right.toArray(new String [right.size()])
							));
				}
				
				// expand left
				if (il >= 0) {
					left.add(0, trans[il]);
					il--;
					
					phones.add(new Polyphone(
							left.toArray(new String [left.size()]), 
							trans[i], 
							right.toArray(new String [right.size()])
							));
				}
			}
			
			left.clear();
			right.clear();
		}
		
		return phones.toArray(new Polyphone [phones.size()]);
	}
	
	/**
	 * Check if the the polyphone and its context matches a transcription at a given point
	 * @param transcription transcription of the target word
	 * @param position index of the central phone
	 * @return true if the polyphone matches the position
	 */
	public boolean matchesTranscription(String [] transcription, int position) {
		// pre-check lengths
		if (left.length > position || right.length >= transcription.length - position)
			return false;
		
		// contexts
		for (int i = 0, j = left.length - 1; i < left.length; ++i, --j) {
			if (!transcription[position - 1 - i].equals(left[j]))
				return false;
		}
		for (int i = 0; i < right.length; ++i) {
			if (!transcription[position + 1 + i].equals(right[i]))
				return false;
		}
		
		// everything fits!
		return true;
	}
}
