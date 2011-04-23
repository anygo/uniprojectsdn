package statistics.jfa;

import framed.MVN;
import io.ChunkedDataSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;

import statistics.MixtureDensity;
import statistics.Sample;
import util.IOUtil;
import Jama.Matrix;
import bin.Mfcc;

public class JFA {

	protected int numSessions = 0, numSpeakers = 0, rank_channel, rank_speaker; // rank
	protected int ev_iterations = 5, ec_iterations = 5;
	// of
	// channel
	// and
	// speaker
	private String statDir = "/home/spsandia104/scheffer/Research/Cep_FA/jfa-fast/accs_m/";
	private String fileDir = "/home/spon6/sachin/SpeakerID/Tasks/SRE05+SRE06/ALL_LISTS/";

	protected ArrayList<String> session_idx; // contains a mapping from the id
												// to
	// the name of the session
	protected ArrayList<String> speaker_idx; // contains a mapping from id to
												// the
	// name of a speaker

	protected Hashtable<String, int[]> ht; // Contains a mapping from a speaker
	// to all available names of
	// sessions;

	// loading matrix
	protected MixtureDensity ubm;

	protected double[] z; // random vector; CF-dim

	protected JFA_Element Ux; // rectangular, low rank; columns are
								// eigenchannels
	protected JFA_Element Vy; // rectangular, low rank; columns are eigenvoices
	// protected Matrix D; // diagonal, CF x CF;
	protected JFA_Element Dz;

	protected boolean verbose = true;
	protected boolean features = true; // indicates if the files in sessions[]
	// are
	// features or not; till now we expect,
	// that the files are MfCCs

	private String audioFormatString = "t:ssg/8";
	private String windowFormatString = "hamm,25,10";
	// this.energyDetect = 5E-7;
	private double energyDetector = 0;
	private String filterbank = "0,4000,-1,.5";
	private String selectionFormatString = "0-11";
	// private String deltaFormatString = "5:1,3:2";
	private String deltaFormatString = "5:1";

	private boolean mvn = true;

	public JFA(MixtureDensity ubm, boolean features) {

		this.ubm = ubm;
		this.features = features;

	}

	public JFA(MixtureDensity ubm, int c_rank, int s_rank,
			String hashtable_File, boolean features) {

		this.features = features;

		this.rank_channel = c_rank;
		this.rank_speaker = s_rank;
		this.ubm = ubm;
		// this.zero_stats = new Matrix(numSessions, ubm.nd);
		// this.first_stats = new Matrix(numSessions, ubm.nd * ubm.fd);
		// this.second_stats = new Matrix(numSessions,ubm.nd * ubm.fd * ubm.fd);

		this.z = new double[ubm.nd * ubm.fd];

		this.Dz = null;

		try {
			buildupHashtable(hashtable_File);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		this.Vy = new JFA_Element(this.numSpeakers, ubm.nd, ubm.fd,
				rank_speaker, this.ev_iterations, "ev", false);
		this.Ux = new JFA_Element(this.numSessions, ubm.nd, ubm.fd,
				rank_channel, this.ec_iterations, "ec", false);

		if (verbose) {
			System.out.println("Feature dimension: " + ubm.fd);
			System.out.println("Number of densities: " + ubm.nd);
			System.out.println("Rank of EC: " + this.rank_channel
					+ " Rank of EV: " + this.rank_speaker);
			// System.out.println(ubm.toString());
		}
	}

	/**
	 * Creates the hashtables wich allows a mapping from the speaker to an array
	 * of all his/her sessions
	 * 
	 * @param fileName
	 *            Name of HashFile
	 * @throws IOException
	 */
	private void buildupHashtable(String fileName) throws IOException {

		if (verbose)
			System.out.println("Output of buildupHashtable");
		this.session_idx = new ArrayList<String>();

		int numSess = 0, numSpeak = 0;
		ht = new Hashtable<String, int[]>();
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String line = "";
		while ((line = br.readLine()) != null) {
			numSpeak++;
			String[] tokens = line.split("\\s");

			if (verbose)
				System.out.print(tokens[0] + ":\t");

			int[] sessions = new int[tokens.length - 1];
			for (int i = 1; i < tokens.length; i++) {
				session_idx.add(tokens[i]);
				sessions[i - 1] = numSess;
				numSess++;
				if (verbose)
					System.out.print(sessions[i - 1] + ","
							+ session_idx.get(i - 1) + " ");
			}
			ht.put(tokens[0], sessions);
			if (verbose)
				System.out.println();
		}

		numSessions = numSess;
		numSpeakers = numSpeak;

		if (verbose)
			System.out.println("Number of speakers: " + numSpeakers
					+ " Number of sessions: " + numSessions);
	}

	public void sumSpeakerStats() {
		System.err.println("sum speaker stats");
		this.speaker_idx = new ArrayList<String>();
		int num_speak = 0;
		for (Enumeration<String> e = ht.keys(); e.hasMoreElements();) {
			String key = e.nextElement();
			speaker_idx.add(key);
			// Iterate over each speaker
			int[] sess = ht.get(key);
			for (int i = 0; i < sess.length; i++) {
				// iterate over the sessions for each speaker, get array-id of
				// session and summarize them

				for (int j = 0; j < ubm.nd; j++) {
					Vy.zero_stats[num_speak][j] += Ux.zero_stats[sess[i]][j];
				}
				for (int j = 0; j < ubm.nd * ubm.fd; j++) {
					Vy.first_stats[num_speak][j] += Ux.first_stats[sess[i]][j];
				}
			}

			/*
			 * for (int j = 0; j < ubm.nd; j++) { Vy.zero_stats[num_speak][j] /=
			 * ht.get(key).length; } for (int j = 0; j < ubm.nd * ubm.fd; j++) {
			 * Vy.first_stats[num_speak][j] /= ht.get(key).length; }
			 */
			num_speak++;
		}
	}

	/**
	 * Fills the zero and first order statistics (zero_stats and first_stats)
	 * Posteriors are calculated in respect to the UBM
	 * 
	 * @throws IOException
	 * @throws FileNotFoundException
	 * 
	 * @throws IOException
	 * @throws Exception
	 */
	public void loadOrComputeStatistics() throws Exception {

		File dir = new File(statDir);

		if ((statDir.isEmpty()) || (!dir.isDirectory())) {
			System.err
					.println("StatDir not specified, compute statistics from scratch");
			computeStatistics();
		} else {
			// Directory exists; go through table and see if statistic files
			System.err.println("Reading statistics from dump directory");

			for (int s = 0; s < numSessions; s++) {
				String zero_stat_file = statDir + session_idx.get(s) + ".zero";
				String first_stat_file = statDir + session_idx.get(s)
						+ ".first";

				System.out.println("reading zero stats: " + zero_stat_file);
				IOUtil.readFloatsFromBinaryFile(zero_stat_file,
						Ux.zero_stats[s], ByteOrder.BIG_ENDIAN);
				IOUtil.readFloatsFromBinaryFile(first_stat_file,
						Ux.first_stats[s], ByteOrder.BIG_ENDIAN);
			}
		}
	}

	public void computeStatisticsOfSession(int sessionIdx) throws Exception {

		String sessionListFile = fileDir + session_idx.get(sessionIdx);
		if (verbose) {
			System.out.println("Processing: " + sessionListFile);
		}
		List<Sample> data = null;

		if (!features) {

			Mfcc mfcc = null;
			int numberOfSamples = 0;
			int num = 0;
			ArrayList<double[]> temp = new ArrayList<double[]>();
			double[] tempMfcc;

			BufferedReader br = new BufferedReader(new FileReader(
					sessionListFile));
			String tempFile = "";

			if (verbose) {
				System.err.println("Reading session files from "
						+ sessionListFile);
			}

			while ((tempFile = br.readLine()) != null) {

				tempFile = tempFile.trim();
				String speechFile = "temp.speech.ssg";

				if (verbose) {

					System.err
							.println("Systemcall:   /home/speech/bocklet/bin/sph2pipe -p "
									+ tempFile + " " + speechFile);
				}
				Runtime.getRuntime().exec(
						"/home/speech/bocklet/bin/sph2pipe -p " + tempFile
								+ " " + speechFile);
				// audioFormatString =
				// RawAudioFormat.getRawAudioFormatFromFile(speechFile).toString();

				if (verbose) {
					System.out.println("Processing speech file:" + speechFile
							+ " / " + audioFormatString);
				}

				File tf = File.createTempFile(Long.toString(System
						.currentTimeMillis())
						+ Double.toString(Math.random()), ".mvn");

				if (mvn) {
					mfcc = new Mfcc(speechFile, audioFormatString,
							windowFormatString, energyDetector, filterbank,
							false, true, selectionFormatString,
							deltaFormatString, null);

					MVN mvn = new MVN();
					mvn.extendStatistics(mfcc);
					mvn.saveToFile(tf.getCanonicalPath());
				}

				mfcc = new Mfcc(speechFile, audioFormatString,
						windowFormatString, energyDetector, filterbank, false,
						true, selectionFormatString, deltaFormatString,
						mvn ? tf.getCanonicalPath() : null);

				tempMfcc = new double[mfcc.getFrameSize()];
				while (mfcc.read(tempMfcc)) {

					// Important, because tempMfcc will be overwritten
					// afterwards
					temp.add(tempMfcc.clone());

					num++;
				}
				if (verbose) {
					System.out.println("Number of Samples from MFCC Object: "
							+ numberOfSamples);
					System.out
							.println("Number of Samples saved into ArrayList: "
									+ num);
				}
			} // end of readList

			double[][] blub = new double[temp.size()][];

			temp.toArray(blub);

			System.err.println("num of features: " + blub.length);
			System.err.println("featDim: " + blub[0].length);

			data = Sample.unlabeledArrayListFromArray(blub);

		} else {
			ChunkedDataSet ds = new ChunkedDataSet(fileDir
					+ session_idx.get(sessionIdx)); // Just
			// read
			data = ds.cachedData();
			// features
			// from file
		}

		double[] temp = new double[ubm.nd];

		// accumulate the statistics
		Iterator<Sample> xiter = data.iterator();
		for (int i = 0; i < data.size(); i++) {
			double[] feat = xiter.next().x;
			ubm.evaluate(feat);
			ubm.posteriors(temp);

			for (int dens = 0; dens < ubm.nd; dens++) {

				Ux.zero_stats[sessionIdx][dens] += temp[dens]; // sum up zero
																// stats

				for (int c = 0; c < feat.length; c++) {

					double val = temp[dens] * feat[c];

					if (!Double.isNaN(val)) {

						Ux.first_stats[sessionIdx][dens * ubm.fd + c] += val; // calculate
						// first
						// order
						// statistics
						// zero_stats

					}
				}
			}
		}
	}

	public void computeStatistics() throws Exception {

		for (int s = 0; s < numSessions; s++) {
			// for all sessions
			computeStatisticsOfSession(s);

		}
	}

	// Input and output functions
	private void writeObject(Object o, String filename) {

		double[] d_temp;
		Matrix m_temp;

		try {
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream(filename));

			if (o instanceof Matrix) {
				if (verbose)
					System.out.println("Writing Matrix object to file");
				m_temp = (Matrix) o;
				oos.writeObject(m_temp);
			}

			if (o instanceof double[]) {
				if (verbose)
					System.out.println("Writing double[] object to file");
				d_temp = (double[]) o;
				oos.writeObject(d_temp);
			}

			oos.flush();
			oos.close();
		} catch (IOException e) {
			System.err.println("Problem writing Object" + o + " to file: "
					+ filename);
			e.printStackTrace();
		}
	}

	public void writeU(String filename) {
		writeObject(Ux, filename);
	}

	public void writeV(String filename) {
		writeObject(Vy, filename);
	}

	public void writeD(String filename) {
		writeObject(Dz, filename);
	}

	public void write_stats() throws IOException {

		if (!statDir.isEmpty()) {

			for (int i = 0; i < numSessions; i++) {
				File f = new File(session_idx.get(i));
				String name = f.getName();
				writeObject(Ux.zero_stats, statDir + File.pathSeparator + name
						+ ".zero_stats");
				writeObject(Ux.first_stats, statDir + File.pathSeparator + name
						+ ".first_stats");
				// writeObject(second_stats,
				// directory+File.pathSeparator+"second_stats");
			}
		} else {
			throw new IOException(
					"Directory for saving statistics not specified");
		}
	}

	/*
	 * public String toString() {
	 * 
	 * String s = "";
	 * 
	 * s += "Speaker Factors (y); [dim: " + Vy.latent.length + "]:\n"; for (int
	 * i = 0; i < Vy.latent.length; i++) { s += "y[i] "; }
	 * 
	 * s += "\n\n Random Vector (z); [dim: " + z.length + "]:\n"; for (int i =
	 * 0; i < z.length; i++) { s += "z[i] "; }
	 * 
	 * s += "\n\n Channel Factors (x); [dim: " + Ux.latent.length + "]:\n"; for
	 * (int i = 0; i < Ux.latent.length; i++) { s += "x[i] "; } s += "\n\n";
	 * return s; }
	 */

	/**
	 * returns Information about the Heap Memory of the JFA
	 */
	public void getMemInfo() {

		Runtime runtime = Runtime.getRuntime();
		int mb = 1024 * 1024;
		System.out.println("##### Heap utilization statistics [MB] #####");
		// Print used memory
		System.out.println("Used Memory:"
				+ (runtime.totalMemory() - runtime.freeMemory()) / mb
				+ " MByte");

		// Print free memory
		System.out.println("Free Memory:" + runtime.freeMemory() / mb
				+ " MByte");

		// Print total available memory
		System.out.println("Total Memory:" + runtime.totalMemory() / mb
				+ " MByte");
		// Print Maximum available memory
		System.out.println("Max Memory:" + runtime.maxMemory() / mb + " MByte");
	}

	public static void main(String[] args) {

		MixtureDensity ubm = null;
		if (args.length != 2) {
			System.err.println("Wrong number of arguments");
			System.exit(-1);
		}
		String ubmFile = args[0];
		String tableFile = args[1];
		try {
			ubm = MixtureDensity.readFromFile(ubmFile);
		} catch (Exception e) {
			System.err.println("Problem reading ubm from file");
			e.printStackTrace();
		}

		JFA jfa = new JFA(ubm, 15, 15, tableFile, false);
		jfa.getMemInfo();
		try {
			jfa.loadOrComputeStatistics();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.err.println("Zero statistics");

		System.out.println(jfa.Ux.zero_stats.length + " "
				+ jfa.Ux.zero_stats[0].length);

		for (int x = 0; x < jfa.Ux.zero_stats.length; x++) {

			for (int y = 0; y < jfa.Ux.zero_stats[x].length; y++) {

				System.out.print(jfa.Ux.zero_stats[x][y] + " ");
			}
			System.out.println();

		}
	}
}
