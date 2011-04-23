package bin;

import sampled.*;
import framed.*;
import io.FrameWriter;
import java.io.*;
import java.util.*;

/**
 * Feature extraction for ASR and Speaker ID. If you change anything, please 
 * increase FEX_VERSION and update LAST_AUTHOR. Feel free to modify CONTRIBUTORS.
 * If you change defaults, do so using the class variables DEFAULT_*
 * 
 * @author sikoried
 *
 */
public class Mfcc implements FrameSource {
	private static final double FEX_VERSION = 1.1;
	private static final String LAST_AUTHOR = "sikoried";
	private static final String CONTRIBUTORS = "sikoried, bocklet, maier, hoenig, steidl";

	/** default audio format: 16 khz, 16 bit, 2b/frame, signed, linear */
	private RawAudioFormat format = new RawAudioFormat();
	
	private AudioSource asource = null;
	private FrameSource window = null;
	private FrameSource pspec = null;
	private FrameSource energy = null;
	private FrameSource melfilter = null;
	private FrameSource dct = null;
	private FrameSource selection = null;
	private FrameSource deltas = null;
	private FrameSource mvn = null;
	
	private FrameSource output = null;
	
	private void initializeAudio(String inFile, String parameterString) throws Exception {
		if (parameterString != null)
			format = RawAudioFormat.create(parameterString);
		
		if (inFile == null || inFile.equals("-"))
			asource = new AudioCapture(format.getBitRate(), format.getSampleRate());
		else if (inFile.startsWith("mixer:"))
			asource = new AudioCapture(inFile.substring(6), (inFile.length() == 6), format.getBitRate(), format.getSampleRate());
		else
			asource = new AudioFileReader(inFile, format, true);
	}
	
	private void initializeWindow(String parameterString) throws Exception {
		window = Window.create(asource, parameterString);
	    output = window;
	}
	
	private void initializeEnergyDetector(double threshold) throws Exception {
		energy = new EnergyDetector(output, threshold);
		output = energy;
	}
	
	private void initializePowerSpectrum() throws Exception {
		pspec = new FFT(output);
		output = pspec;
	}
	
	private void initializeMelfilter(String parameterString) throws Exception {
		melfilter = Melfilter.create(output, format.getSampleRate(), parameterString);
		output = melfilter;
	}
	
	private void initializeDCT(boolean doShortTimeEnergy) throws Exception {
		dct = new DCT(output, true, doShortTimeEnergy);
		output = dct;
	}
	
	private void initializeSelection(String parameterString) throws Exception {
		if (parameterString == null)
			selection = new Selection(output);
		else
			selection = Selection.create(output, parameterString);
		
		output = selection;
	}
	
	private void initializeDeltas(String parameterString) throws Exception {
		if (parameterString == null)
			return;
		
		deltas = Slope.create(output, parameterString);
		output = deltas;
	}
	
	private void initializeMVN(String parameterFile) throws Exception {
		mvn = new MVN(output, parameterFile);
		output = mvn;
	}
	
	@Deprecated
	private void initializeCMS(String cmsParamFile) throws Exception {
		output = CMS1.loadMeanFromFile(cmsParamFile, output);
	}
	
	/**
	 * Initialize the new MFCC object using the given parameter strings. If a 
	 * parameter String is null, the default constructor is called, or nthe object
	 * is not integrated in the pipe line (deltas, CMS)
	 * @param inFile file name to open
	 * @param pAudio Audio format parameter string, e.g. t:ssg/16
	 * @param pWindow Window function to use, e.g. hamm,25,10
	 * @param energyDetect Energy threshold to filter out non-speech (non-activity) frames
	 * @param pFilterbank Mel filter bank parameters, e.g. 0,8000,-1,.5
	 * @param onlySpectrum Flag if the cepstrum computation should be EXCLUDED
	 * @param doShortTimeEnergy Flag to include the short time band energy (instead of the 0th coefficient)
	 * @param pSelection Perform a selection on the feature vector (usually 0-11)
	 * @param pDeltas Derivatives to compute, e.g. 1:5,2:3
	 * @param mvnParamFile Parameter file for mean and variance normalization (if applicable)
	 * @throws Exception
	 */
	public Mfcc(String inFile, String pAudio, String pWindow, double energyDetect, 
			String pFilterbank, boolean onlySpectrum, boolean doShortTimeEnergy, 
			String pSelection, String pDeltas, String mvnParamFile) 
		throws Exception {
		initializeAudio(inFile, pAudio);
		initializeWindow(pWindow);
		initializePowerSpectrum();
		
		if(energyDetect >0)
			initializeEnergyDetector(energyDetect);
		if (pFilterbank != null)
			initializeMelfilter(pFilterbank);
		if (!onlySpectrum)
			initializeDCT(doShortTimeEnergy);
		
		initializeSelection(pSelection);

		if (pDeltas != null)
			initializeDeltas(pDeltas);
		
		if (mvnParamFile != null) 
			initializeMVN(mvnParamFile);
	}
	
	/**
	 * Initialize the new MFCC object using the given parameter strings. If a 
	 * parameter String is null, the default constructor is called, or nthe object
	 * is not integrated in the pipe line (deltas, CMS)
	 * @param inFile file name to open
	 * @param pAudio Audio format parameter string, e.g. t:ssg/16
	 * @param pWindow Window function to use, e.g. hamm,25,10
	 * @param energyDetect Energy threshold to filter out non-speech (non-activity) frames
	 * @param pFilterbank Mel filter bank parameters, e.g. 0,8000,-1,.5
	 * @param onlySpectrum Flag if the cepstrum computation should be EXCLUDED
	 * @param doShortTimeEnergy Flag to include the short time band energy (instead of the 0th coefficient)
	 * @param pSelection Perform a selection on the feature vector (usually 0-11)
	 * @param pDeltas Derivatives to compute, e.g. 1:5,2:3
	 * @param CMS parameter file
	 * @param dummy argument, not used!
	 * @throws Exception
	 */
	@Deprecated
	public Mfcc(String inFile, String pAudio, String pWindow, double energyDetect, 
			String pFilterbank, boolean onlySpectrum, boolean doShortTimeEnergy, 
			String pSelection, String pDeltas, String cmsParamFile, Object dummy) 
		throws Exception {
		initializeAudio(inFile, pAudio);
		initializeWindow(pWindow);
		initializePowerSpectrum();
		
		if(energyDetect >0)
			initializeEnergyDetector(energyDetect);
		if (pFilterbank != null)
			initializeMelfilter(pFilterbank);
		if (!onlySpectrum)
			initializeDCT(doShortTimeEnergy);
		
		initializeSelection(pSelection);

		if (cmsParamFile != null) 
			initializeCMS(cmsParamFile);
		
		if (pDeltas != null)
			initializeDeltas(pDeltas);
			
	}
	
	public String describePipeline() {
		StringBuffer buf = new StringBuffer();
		buf.append(asource + "\n");
		buf.append(window + "\n");
		buf.append(pspec + "\n");
		buf.append(melfilter + "\n");
		buf.append(dct + "\n");
		buf.append(selection + "\n");
		if (deltas != null)
			buf.append(deltas + "\n");
		if (mvn != null) 
			buf.append(mvn + "\n");
		return buf.toString();
	}
	
	public void tearDown() throws IOException {
		asource.tearDown();
	}
	
	public boolean read(double [] buf) throws IOException {
		return output.read(buf);
	}
	
	public int getFrameSize() {
		return output.getFrameSize();
	}

	/** 16kHz, 16bit, signed, little endian, linear */
	public static String DEFAULT_AUDIO_FORMAT = "t:ssg/16";
	
	/** Hamming window of 16ms, 10ms shift */
	public static String DEFAULT_WINDOW = "hamm,16,10";
	
	/** Filter bank 188Hz-6071Hz, 226.79982mel band width, 50% filter overlap */
	public static String DEFAULT_MELFILTER = "188,6071,226.79982,0.5";
	
	/** Deltas to compute (null = none) */
	public static String DEFAULT_DELTAS = null;
	
	/** Static features to select after DCT */
	public static String DEFAULT_SELECTION = "0-11";
	
	/** Program synopsis */
	private static final String SYNOPSIS = 
		"mfcc feature extraction v " + FEX_VERSION + "\n" +
		"last author: " + LAST_AUTHOR + "\n" +
		"contributors: " + CONTRIBUTORS + "\n\n" +
		"usage: bin.Mfcc [options]\n\n" +
		"file options:\n\n" +
		"-i in-file\n" +
		"  use the given file for input; use \"-i -\" for default microphone\n" +
		"  input, or \"mixer:mixer-name\" for (and only for) a specific mixer\n" +
		"-o out-file\n" +
		"  use the given file for output (header + double frames; default: STDOUT)\n" +
		"--in-out-list listfile\n" +
		"  the list contains lines \"<in-file> <out-file>\" for batch processing\n" +
		"--in-list listfile directory\n" +
		"  contains lines \"<file>\" for input; strips directory from input files," +
		"  and stores them in <directory>\n" +
		"--ufv\n" +
		"  write UFVs instead of header + double frames\n" +
		"\n" +
		"audio format options:\n\n" +
		"-f <format-string>\n" +
		"  \"f:path-to-file-with-header\": load audio format from file\n" +
		"  \"t:template-name\": use an existing template (ssg/[8,16], alaw/[8,16], ulaw/[8,16]\n" +
		"  \"r:bit-rate,sample-rate,signed(0,1),little-endian(0,1)\": specify raw format (no-header)\n" +
		"  default: \"" + DEFAULT_AUDIO_FORMAT + "\"\n" +
		"\n" +
		"feature extraction options:\n\n" +
		"-w \"<hamm|hann|rect>,<length-ms>,<shift-ms>\"" +
		"  window function (Hamming, Hann, Rectangular), length of window and \n" +
		"  shift time (in ms)\n" +
		"  default: \"" + DEFAULT_WINDOW + "\"\n" +
		"--no-filterbank\n" +
		"  Do NOT apply a filterbank at all\n" +
		"-b \"<startfreq-hz>,<endfreq-hz>,<bandwidth-mel>,<val>\"\n" +
		"  mel filter bank; val < 1. : minimum overlap <val>; val > 1 : minimum\n" +
		"  number of filters; if you wish to leave a certain field at default,\n" +
		"  put a value below 0.\n" +
		"  default: \"" + DEFAULT_MELFILTER + "\"\n" +
		"--no-short-time-energy\n" +
		"  Do NOT compute the short time energy, use the 0th cepstral coefficient instead\n" +
		"--only-spectrum\n" +
		"  Do NOT apply DCT after the filtering\n" +
		"-s <selection-string>\n" +
		"  Select the static features to use and in which order, e.g. \"0,3-8,1\"\n" +
		"  default: \"" + DEFAULT_SELECTION + "\"\n" +
		"-m <mvn-file>\n" +
		"  use statistics saved in <mvn-file> for mean and variance normalization (MVN)\n" +
		"--generate-mvn-file <mvn-file>\n" +
		"  computes mean and variance statistics on the given file(list) and saves\n" +
		"  it to <mvn-file>\n" +
		"--turn-wise-mvn\n" +
		"  Apply MVN to each turn; this is an individual offline mean and variance\n" +
		"  normalization\n" +
		"-d \"context:order[:scale][,context:order[:scale]]+\"\n" +
		"  compute oder <order> derivatives over context <context>, and optionally\n" +
		"  scale by <scale>, separate multiple derivatives by comma; deltas are\n" +
		"  concatenated to static features in the same order as specified in the\n" +
		"  argument;\n" + 
		"  default: \"" + DEFAULT_DELTAS + "\"\n" +
		"-e \"threshold\"n" +
		"  perform an energy (voice activity) detection; \n" +
		"  all frames with an energy lower than <threshold> will be removed\n" +
		"--calculate-energy-threshold\n"+
		"  calculates the energy threshold; use audio files with silence for this"+
		"\n" +
		"-h | --help\n" +
		"  display this help text\n\n" +
		"--show-pipeline\n" +
		"  initialize and print feature pipeline to STDERR\n";
	
	public static void main(String[] args) throws Exception {
		// defaults
		boolean displayHelp = false;
		boolean showPipeline = false;
		boolean ufv = false;
		
		boolean generateMVNFile = false;
		boolean calculateEnergyThreshold = false;
		
		boolean turnwisemvn = false;
		boolean noFilterbank = false;
		boolean onlySpectrum = false;
		boolean doShortTimeEnergy = true;
		
		String inFile = null;
		String outFile = null;
		String outDir = null;
		String listFile = null;
		String mvnParamFile = null;
		
		double energyDetector = 0.00;
		
		String audioFormatString = DEFAULT_AUDIO_FORMAT;
		String windowFormatString = DEFAULT_WINDOW;
		String filterFormatString = DEFAULT_MELFILTER;
		String selectionFormatString = DEFAULT_SELECTION;
		String deltaFormatString = DEFAULT_DELTAS;
		
		if (args.length > 1) {
			// process arguments
			for (int i = 0; i < args.length; ++i) {
				
				// file options
				if (args[i].equals("--in-out-list"))
					listFile = args[++i];
				else if (args[i].equals("-i"))
					inFile = args[++i];
				else if (args[i].equals("-o"))
					outFile = args[++i];
				else if (args[i].equals("--ufv"))
					ufv = true;
				else if (args[i].equals("--in-list")) {
					listFile = args[++i];
					outDir = args[++i];
				}
				
				// audio format options
				else if (args[i].equals("-f"))
					audioFormatString = args[++i];
				
				// window options
				else if (args[i].equals("-w")) 
					windowFormatString = args[++i];
				
				// mel filter bank options
				else if (args[i].equals("-b"))
					filterFormatString = args[++i];
				
				// selection?
				else if (args[i].equals("-s"))
					selectionFormatString = args[++i];
				
				// mean options
				else if (args[i].equals("-m"))
					mvnParamFile = args[++i];
				else if (args[i].equals("--generate-mvn-file")) {
					generateMVNFile = true;
					mvnParamFile = args[++i];
				} else if (args[i].equals("--turn-wise-mvn")) 
					turnwisemvn = true;
				else if (args[i].equals("--no-filterbank"))
					noFilterbank = true;
				else if (args[i].equals("--only-spectrum"))
					onlySpectrum = true;
				else if (args[i].equals("--no-short-time-energy"))
					doShortTimeEnergy = false;
				
				// deltas
				else if (args[i].equals("-d"))
					deltaFormatString = args[++i];
				
				// energy detection
				else if (args[i].equals("-e")) {
					energyDetector = Double.valueOf(args[++i]);
					//System.err.println(energyDetector);
					}

				else if (args[i].equals("--calculate-energy-threshold"))
					calculateEnergyThreshold = true;
				
				// help?
				else if (args[i].equals("-h") || args[i].equals("--help"))
					displayHelp = true;
				
				// show pipeline?
				else if (args[i].equals("--show-pipeline"))
					showPipeline = true;
				
				// whoops...
				else
					System.err.println("ignoring argument " + i + ": " + args[i]);
			}
		} else {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		// help?
		if (displayHelp) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		// consistency checks
		if (listFile != null && (inFile != null || outFile != null))
			throw new Exception("-l and (-i,-o) are exclusive!");
		if (turnwisemvn && mvnParamFile != null)
			throw new Exception("--generate-mvn-file, -m and --turnwise-mvn are exclusive");
		
		ArrayList<String> inlist = new ArrayList<String>();
		ArrayList<String> outlist = new ArrayList<String>();
		
		// read list
		if (listFile == null) {
			inlist.add(inFile);
			outlist.add(outFile);
		} else {
			BufferedReader lr = new BufferedReader(new FileReader(listFile));
			String line = null;
			int i = 1;
			while ((line = lr.readLine()) != null) {
				if (outDir == null) {
					String [] help = line.split("\\s+");
					if (help.length != 2)
						throw new Exception("file list is broken at line " + i);
					inlist.add(help[0]);
					outlist.add(help[1]);
				} else {
					String [] help = line.split("/");
					inlist.add(line);
					outlist.add(outDir + "/" + help[help.length-1]);
				}
				i++;
			}
		}
		
		Mfcc mfcc = null;
		
		// There is two kind of jobs which only require to process all data but
		// no feature write out.
		if (calculateEnergyThreshold || generateMVNFile) {
			double etr = 0.;
			MVN mvn = new MVN();
			
			while (inlist.size() > 0) {
				if (calculateEnergyThreshold) {
					AudioSource asource = new AudioFileReader(inFile,RawAudioFormat.create(audioFormatString) , true);
					FrameSource window = Window.create(asource, windowFormatString);
					etr += EnergyDetector.calcThresholdFromSilence(window);
				} else if (generateMVNFile){
					mfcc = new Mfcc(inFile, audioFormatString, windowFormatString, 
							energyDetector, noFilterbank ? null : filterFormatString, 
							onlySpectrum, doShortTimeEnergy, selectionFormatString, null, null);
					
					mvn.extendStatistics(mfcc);
				} else 
					throw new Exception("Reached unreachable state :(");
			}
			
			if (calculateEnergyThreshold) {
				System.out.println("Calculated Energy Threshold:" + etr);
				return;
			} else if (generateMVNFile){
				System.out.println("Saving mean and variance statistics to " + mvnParamFile);
				mvn.saveToFile(mvnParamFile);
			} else 
				throw new Exception("Reached unreachable state :(");
			
			// we're done here!
			System.exit(0);
		}
		
		// if we do a turn-wise cms, we need a temporary file
		File tf = null;
		if (turnwisemvn) 
			tf = File.createTempFile(Long.toString(System.currentTimeMillis()) + Double.toString(Math.random()), ".mvn");
		
		// Do the actual feature computation and write out
		while (inlist.size() > 0) {
			// get next file
			inFile = inlist.remove(0);
			outFile = outlist.remove(0);
			
			// if there is turn-wise MVN, we need to compute the statistics first!
			if (turnwisemvn) {
				mfcc = new Mfcc(inFile, audioFormatString, windowFormatString, 
						energyDetector, noFilterbank ? null : filterFormatString, 
						onlySpectrum, doShortTimeEnergy, selectionFormatString, 
						deltaFormatString, null);
				
				MVN mvn = new MVN();
				mvn.extendStatistics(mfcc);
				mvn.saveToFile(tf.getCanonicalPath());
			} 
			
			// regular processing: if there's (temporal) MVN data, it's applied
			mfcc = new Mfcc(inFile, audioFormatString, windowFormatString, 
					energyDetector, noFilterbank ? null : filterFormatString, 
					onlySpectrum, doShortTimeEnergy, selectionFormatString, 
					deltaFormatString, turnwisemvn ? tf.getCanonicalPath() : mvnParamFile);
			
			double [] buf = new double [mfcc.getFrameSize()];
			
			FrameWriter writer = new FrameWriter(buf.length, outFile, ufv);
			while (mfcc.read(buf)) 
				writer.write(buf);
			
			writer.close();
			
			// clean up the temporary file
			if (turnwisemvn)
				tf.delete();
		}

		// output pipeline?
		if (showPipeline)
			System.err.print(mfcc.describePipeline());
	}
}
