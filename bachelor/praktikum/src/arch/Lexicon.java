package arch;

import exceptions.InvalidFormatException;
import exceptions.OutOfVocabularyException;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Collections;

public class Lexicon implements Serializable {
	private static final long serialVersionUID = 1L;

	/** the phonetic alphabet (monophones) */
	private ArrayList<String> alphabet;
	
	/** lexicon entries, supposedly sorted */
	public LinkedList<Entry> entries = new LinkedList<Entry>();
	
	/**
	 * Generate a Lexicon using the given phonetic alphabet.
	 * @param alphabet list of Strings, representing the alphabet
	 */
	public Lexicon(ArrayList<String> alphabet) {
		this.alphabet = alphabet;
		
	}
	
	/**
	 * Use the lexicon to obtain a transcription for a given word. Does a binary
	 * search on the lexicon list.
	 * 
	 * @param word requested word
	 * @return transcription as String array
	 * @throws OutOfVocabularyException
	 */
	public String [] translate(String word) 
		throws OutOfVocabularyException {
		int pos = Collections.binarySearch(entries, new Entry(word, null));
		
		if (pos < 0)
			throw new OutOfVocabularyException(word);
		
		return entries.get(pos).transcription;
	}
	
	/**
	 * Get a String representation of the Lexicon
	 */
	public String toString() {
		return "Lexicon size_alphabet=" + alphabet.size() + " size_lexicon=" + entries.size();
	}
	
	/**
	 * Insert an entry to the lexicon and sort it
	 * @param e
	 */
	public void insertEntrySorted(Entry e) {
		entries.add(e);
		Collections.sort(entries);
	}
	
	/**
	 * Insert an entry without sorting the lexicon afterwards
	 * @param e
	 */
	public void insertEntry(Entry e) {
		entries.add(e);
	}
	
	/**
	 * Sort the lexicon
	 */
	public void sortEntries() {
		Collections.sort(entries);
	}
	
	/** 
	 * Retrieve the corresponding Lexicon entry for the given word
	 * @param word
	 * @return
	 */
	public Entry getEntry(String word) {
		for (Entry e : entries)
			if (e.word.equals(word))
				return e;
		return null;
	}
	
	/**
	 * Add an entry using the word and its transcription.
	 * @param word
	 * @param transcription
	 * @throws InvalidFormatException on error, however file and line number are wrong
	 */
	public void addEntry(String word, String transcription) 
		throws InvalidFormatException {
		ArrayList<String> trans = new ArrayList<String>();
		
		// split to phoneme array
		for (int pos = 0; pos < transcription.length(); ++pos) {
			// 2-char phoneme?
			if (pos < transcription.length() - 1) {
				String m = transcription.substring(pos, pos+2);
				if (alphabet.contains(m)) {
					trans.add(m);
					pos++;
					continue;
				}
			}
			// 1-char phoneme?
			String m = transcription.substring(pos, pos + 1);
			if (alphabet.contains(m) || m.equals(Polyphone.SB) || m.equals(Polyphone.WB)) {
				trans.add(m);
				continue;
			}
			
			// whoops...
			throw new InvalidFormatException(null, -1, "invalid character '" + m + "' at position " + pos);
		}
		
		// add word boundaries at beginning and end
		trans.add(0,Polyphone.WB);
		trans.add(Polyphone.WB);
		
		// generate and add entry
		insertEntry(new Entry(word, trans.toArray(new String [trans.size()])));
	}
	
	public static Lexicon readLexiconFromFile(String alphabetFile, String lexiconFile) 
		throws InvalidFormatException, IOException {
		String buf;
		
		// first, read in the alphabet
		BufferedReader br = new BufferedReader(new FileReader(alphabetFile));
		ArrayList<String> alphabet = new ArrayList<String>();
		
		while ((buf = br.readLine()) != null)
			alphabet.add(buf);
		
		br.close();
		
		// create the lexicon
		Lexicon lex = new Lexicon(alphabet);
		
		// now read in the lexicon from the file
		int lno = 0;
		br = new BufferedReader(new FileReader(lexiconFile));
		
		while ((buf = br.readLine()) != null) {
			lno++;
			
			int pos = buf.indexOf("\t");
			if (pos == -1)
				throw new InvalidFormatException(lexiconFile, lno, "not a vald line");
			
			try {
				lex.addEntry(buf.substring(0, pos), buf.substring(pos+1));
			} catch (InvalidFormatException e) {
				e.lineNumber = lno;
				e.fileName = lexiconFile;
				throw e;
			}
		}
		
		// sort all entries
		lex.sortEntries();
		
		return lex;
	}
	
	/**
	 * The Entry class stores a lexicon entry by its word and transcription
	 * 
	 * @author sikoried
	 */
	public static class Entry implements Comparable <Entry>, Serializable {
		private static final long serialVersionUID = 1L;

		/** The actual word */
		public String word;
		
		/** The transcription using literals from Lexicon.alphabet including word boundaries at beginning end end*/
		public String [] transcription;
		
		/**
		 * Create a new lexicon entry using given word and transcription
		 * @param word
		 * @param transcription
		 */
		public Entry(String word, String [] transcription) {
			this.word = word;
			this.transcription = transcription;
		}
		
		/**
		 * lexical sort
		 */
		public int compareTo(Entry e) {
			return word.compareTo(e.word);
		}
		
		public boolean equals(Object o) {
			if (o instanceof Entry)
				return equals((Entry) o);
			return false;
		}
		
		/**
		 * Returns true if the words match
		 * @param e
		 * @return
		 */
		public boolean equals(Entry e) {
			return e.word.equals(word);
		}
		
		public String toString() {
			StringBuffer sb = new StringBuffer();
			sb.append(word + "\t");
			for (String t : transcription)
				sb.append(t);
			return sb.toString();
		}
	}
	
	private static final String synopsis = 
		"usage: arch.Lexicon alphabet-file-1 lexicon-file-1 [<alphabet-file-2 lexicon-file-2> ...]\n" +
		"\n" +
		"Use this program to verify a lexicon alongside with its alphabet\n";
	
	/**
	 * Use the main program to verify a lexicon alongside with its alphabet
	 * @param args
	 */
	public static void main(String [] args) {
		if (args.length < 2 || args.length % 2 == 1) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		for (int i = 0; i < args.length; ++i) {
			String file_a = args[i++];
			String file_l = args[i];
			
			try {
				System.out.println("CALL readLexiconFromFile(\"" + file_a + "\", \"" + file_l + "\")");
				Lexicon lex = readLexiconFromFile(file_a, file_l);
				System.out.println(lex);
			} catch (InvalidFormatException e) {
				System.err.println(e);
			} catch (IOException e) {
				System.err.println(e);
			}
		}
	}
}
