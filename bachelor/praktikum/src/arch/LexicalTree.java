package arch;

import java.util.*;
import java.io.*;
import exceptions.*;
import statistics.HMM;
import util.Pair;

/**
 * The LexicalTree stores the compiled lexicon. The tree is organized by phone
 * prefixes and used for both training and recognition.
 *  
 * @author sikoried
 *
 */
public class LexicalTree implements Serializable {
	private static final long serialVersionUID = 1L;

	/** node instance counter */
	private static long node_instances = 0;
	
	/**
	 * Helper class to build up a tree. Each node stores the phone (for the 
	 * acoustic parameters), its parent and children. If the end of the word
	 * is reached, the lexical entry is non-null.
	 * 
	 * @author sikoried
	 *
	 */
	class TreeNode implements Serializable {
		private static final long serialVersionUID = 1L;

		/** Associated acoustic information */
		Polyphone phone = null;
		
		/** If this is a leaf, there will be a word -- otherwise null */
		Lexicon.Entry word = null;
		
		TreeNode parent;
		TreeNode [] children = new TreeNode [0];
		
		TreeNode(Polyphone phone, TreeNode parent) {
			this.phone = phone;
			this.parent = parent;
		}
		
		/** 
		 * determine the start node for this word (required at training time)
		 * @return origin of this node
		 */
		public TreeNode startNode() {
			TreeNode n = parent;
			while (n.parent != root)
				n = n.parent;
			return n;
		}
		
		/** unique instance id */
		public final long INSTANCE_ID = node_instances++;
		
		/** return a string representation of this node */
		public String toString() {
			if (phone == null)
				return "<null>";
			else if (word == null)
				return phone.toString();
			else
				return phone.toString() + " WORD_BOUNDARY " + word.word;
		}
	}
	
	/** The root of the tree to reach all words */
	private TreeNode root = new TreeNode(null, null);
	
	/** The available phones */
	private PhoneInventory phoneInventory;
	
	/** available words */
	private Lexicon lexicon;
	
	/** translation table for word -> leaf in the lexical tree */
	private HashMap<String, TreeNode> wordLeaves = new HashMap<String, TreeNode>();
	
	/**
	 * Construct a lexical tree using the given PhoneInventory and Lexicon.
	 * @param phoneInventory
	 * @param lexicon
	 */
	public LexicalTree(PhoneInventory phoneInventory, Lexicon lexicon) {
		this.phoneInventory = phoneInventory;
		this.lexicon = lexicon;
		
		rebuildTree();
	}
	
	/** 
	 * Rebuild the lexical tree structure using the current PhoneInventory and
	 * Lexicon
	 */
	public void rebuildTree() {
		// delete the old tree
		root = new TreeNode(null, null);
		
		// add all words from the lexicon
		for (Lexicon.Entry entry : lexicon.entries)
			addToTree(entry, phoneInventory.translateWord(entry.transcription));
	}
	
	/**
	 * Add a Polyphone sequence (i.e. a word) to the lexical tree
	 * @param transcription
	 */
	public void addToTree(Lexicon.Entry word, Polyphone [] transcription) {
		addToTree(word, root, transcription, 0);
	}
	
	/**
	 * Internal recursive function to add a word. Very similar to the PhoneInventory
	 * tree structure.
	 * 
	 * @param node node under inspection
	 * @param trans transcription
	 * @param ind Polyphone under inspection
	 */
	private void addToTree(Lexicon.Entry word, TreeNode node, Polyphone [] trans, int ind) {
		// search if we already have this polyphone
		TreeNode result = null;
		int i = 0;
		for (i = 0; i < node.children.length; ++i) {
			if (node.children[i].phone.equals(trans[ind])) {
				result = node.children[i];
				break;
			}
		}
		
		// nothing found, add a new child
		if (result == null) {
			TreeNode [] nc = new TreeNode [node.children.length + 1];
			System.arraycopy(node.children, 0, nc, 0, node.children.length);
			result = new TreeNode(trans[ind], node); 
			nc[node.children.length] = result;
			node.children = nc;
		} 
		
		// see if we need more recursion; if not, mark as word boundary and 
		// update the translation table
		if (ind == trans.length - 1) {
			result.word = word;
			wordLeaves.put(word.word, result);
		} else
			addToTree(word, result, trans, ind + 1);
	}
	
	/**
	 * Synthesize a sequence of HMMs for the given sentence
	 * @param sent List of Lexicon.Entrys representing the sentence
	 * @return sequence of hmms
	 */
	public HMM [] synthesize(Iterable<Lexicon.Entry> sent)
		throws OutOfVocabularyException {
		ArrayList<HMM> hmms = new ArrayList<HMM>();
		
		for (Lexicon.Entry w : sent) {
			TreeNode leaf = wordLeaves.get(w.word);
			
			// follow the leaf to the root, build up the acoustic model
			LinkedList<HMM> models = new LinkedList<HMM>();
			while (leaf.parent != null) {
				models.addFirst(leaf.phone.hmm);
				leaf = leaf.parent;
			}
			
			// add the model to the big list
			hmms.addAll(models);
		}
		
		return hmms.toArray(new HMM [hmms.size()]);
	}
	
	/**
	 * Return a String representation of the lexical tree.
	 */
	public String toString() {
		return "LexicalTree: " + wordLeaves.keySet().size() + " words";
	}
	
	public String treeAsString() {
		StringBuffer sb = new StringBuffer();
		ArrayList<Pair<Integer, TreeNode>> agenda = new ArrayList<Pair<Integer, TreeNode>>();
		agenda.add(new Pair<Integer, TreeNode>(0, root));
		final String INDENT = "    ";

		// depth first search
		while (agenda.size() > 0) {
			Pair<Integer, TreeNode> pair = agenda.remove(agenda.size() - 1);

			// do correct indent
			for (int i = 0; i < pair.a; ++i)
				sb.append(INDENT);

			// print current
			sb.append(pair.b.toString());

			// add the children
			for (TreeNode child : pair.b.children)
				agenda.add(new Pair<Integer, TreeNode>(pair.a + 1, child));

			// finish line
			sb.append("\n");
		}
		return sb.toString();
	}
	
	/**
	 * Return a .dot representation of the LexicalTree for use with a graph plotter
	 * @param includeHeader include the .dot header?
	 * @return
	 */
	public String treeAsDotFormat(boolean includeHeader) {
		// iterative approach
		ArrayList<TreeNode> agenda = new ArrayList<TreeNode>();
		agenda.add(root);
		
		StringBuffer sb = new StringBuffer();
		
		if (includeHeader) {
			sb.append("digraph LexicalTree {\n" +
				"ordering=out;\n" +
				"rankdir=LR;\n" +
				"node [shape=box];\n");
		}
		
		// DFS search
		while (agenda.size() > 0) {
			TreeNode p = agenda.remove(agenda.size() - 1);
			sb.append("node_" + p.INSTANCE_ID + " [label=\"" + p.toString() + "\"];\n");
			if (p.parent != null)
				sb.append("node_" + p.INSTANCE_ID + " -> node_" + p.parent.INSTANCE_ID + ";\n");
			for (TreeNode child : p.children)
				agenda.add(child);
		}
		
		if (includeHeader)
			sb.append("}\n");
		
		return sb.toString();
	}
	
	public static final String synopsis = 
		"sikoried, 12-15-2009\n\n" +
		"Construct and/or view a lexical tree required for training and recognition.\n\n" +
		"usage: arch.LexicalTree [options]\n" +
		"  --load file\n" +
		"    Load a pre-compiled lexical tree from the given file.\n" +
		"  --construct alphabet lexicon phone-inv out-file\n" +
		"    Construct a new lexical tree for the given lexicon and phone inventory\n" +
		"    and save it to the given out-file.\n" +
		"\n" +
		"  --print-all\n" +
		"    Print the complete hierarchy\n" +
		"  --list-words\n" +
		"    List all the words and their transcriptions (in alphabetical order)\n" +
		"  --find word\n" +
		"    Check if a word exists in the tree and show its transcription. Use\n" +
		"    a comma separated list if you want to query more than one word.\n" +
		"  --dot-format\n" +
		"    Use .dot format instead of human readable ASCII format.\n";
	
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		if (args.length < 2) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		String tFile = null;
		String aFile = null;
		String lFile = null;
		String pFile = null;
		String oFile = null;
		
		boolean printAll = false;
		boolean listWords = false;
		boolean dotFormat = false;

		ArrayList<String> findWord = new ArrayList<String>();
		
		for (int i = 0; i < args.length; ++i) {
			if (args[i].equals("--load"))
				tFile = args[++i];
			else if (args[i].equals("--construct")) {
				aFile = args[++i];
				lFile = args[++i];
				pFile = args[++i];
				oFile = args[++i];
			} else if (args[i].equals("--print-all"))
				printAll = true;
			else if (args[i].equals("--list-words"))
				listWords = true;
			else if (args[i].equals("--find")) {
				String [] words = args[++i].split(",");
				for (String w : words)
					findWord.add(w);
			} else if (args[i].equals("--dot-format"))
				dotFormat = true;
			else
				throw new IOException("Invalid argument \"" + args[i] + "\"");
		}
		
		LexicalTree lt = null;
		
		if (tFile != null) {
			System.err.print("Loading lexical tree from \"" + tFile + "\"...");
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(tFile));
			lt = (LexicalTree) ois.readObject();
			ois.close();
			System.err.println("OK");
		} else if (aFile != null && lFile != null && pFile != null && oFile != null) {
			System.err.print("Loading lexicon...");
			Lexicon lex = Lexicon.readLexiconFromFile(aFile, lFile);
			System.err.println("OK\n" + lex);
			
			System.err.print("Reading PhoneInventory...");
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(pFile));
			PhoneInventory pi = (PhoneInventory) ois.readObject();
			ois.close();
			System.err.println("OK\n" + pi);
			
			System.err.print("Constructing lexical tree...");
			lt = new LexicalTree(pi, lex);
			System.err.println("OK\n" + lt);
		} else 
			throw new IOException("Specify either --load or --construct.");
		
		if (oFile != null) {
			System.err.print("Writing lexical tree to \"" + oFile + "\"...");
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(oFile));
			oos.writeObject(lt);
			oos.close();
			System.err.println("OK");
		}
		
		if (printAll) {
			System.out.println(dotFormat ? lt.treeAsDotFormat(true) : lt.treeAsString());
		}
		
		if (listWords) {
			LinkedList<String> leafs = new LinkedList<String>(lt.wordLeaves.keySet());
			Collections.sort(leafs);
			for (String l : leafs) {
				TreeNode n = lt.wordLeaves.get(l);
				System.out.print(n.word.word);
				ArrayList<Polyphone> toreverse = new ArrayList<Polyphone>();
				toreverse.add(n.phone);
				while (n.parent != null && n.parent.phone != null) {
					n = n.parent;
					toreverse.add(n.phone);
				}
				while (toreverse.size() > 0)
					System.out.print(" " + toreverse.remove(toreverse.size() - 1));
				System.out.println();
			}
		}
		
		for (String w : findWord) {
			TreeNode n = lt.wordLeaves.get(w);
			if (n == null)
				System.out.println("Word \"" + w + "\" not found");
			else {
				System.out.print(n.word.word);
				ArrayList<Polyphone> toreverse = new ArrayList<Polyphone>();
				toreverse.add(n.phone);
				while (n.parent != null && n.parent.phone != null) {
					n = n.parent;
					toreverse.add(n.phone);
				}
				while (toreverse.size() > 0)
					System.out.print(" " + toreverse.remove(toreverse.size() - 1));
				System.out.println();
			}
		}
		
	}
}
