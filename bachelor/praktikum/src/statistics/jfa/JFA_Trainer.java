package statistics.jfa;

import java.io.IOException;
import java.io.PrintWriter;

import Jama.Matrix;

import statistics.*;

public class JFA_Trainer extends JFA {

	/**
	 * Constructor of JFA_Trainer. Create an JFA Object and Calculate zero and
	 * first order statistics
	 * 
	 * @param ubm
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */

	private Matrix independent[] = null;
	private double[] iSupervector = null;

	/**
	 * Initialize JFA_Trainer by reading a UBM (GMM object) from file
	 * 
	 * @param ubmFile
	 * @param tableFile
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	JFA_Trainer(String ubmFile, String tableFile) throws IOException,
			ClassNotFoundException {

		super(MixtureDensity.readFromFile(ubmFile), 100, 200, tableFile, false);
		iSupervector = new double[ubm.nd * ubm.fd];
		independent = new Matrix[ubm.nd];
	}

	/**
	 * Initialize JFA_Trainer by reading UBM (cova und mean) from ASCII files
	 * 
	 * @param meanFile
	 * @param covaFile
	 * @param tableFile
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	JFA_Trainer(String meanFile, String covaFile, String tableFile)
			throws IOException, ClassNotFoundException {

		super(MixtureDensity.fillMDFromASCII(39, 128, meanFile, covaFile), 100,
				200, tableFile, false);
		iSupervector = new double[ubm.nd * ubm.fd];
		independent = new Matrix[ubm.nd];

	}

	/**
	 * train the Eigenvoice Matrix; D is set to zero in this case; The
	 * eigenvoice matrix is initialized with String init; if init == null a new
	 * randomMatrix is used for initialization
	 * 
	 * @throws IOException
	 */
	public void trainEigenvoiceMatrix(String init) throws IOException {
		// according to Kenny08-ASO D can be initialized with 0; V should be
		// initialized randomly

		Vy.factors = new double[Vy.num][this.ec_iterations];
		
		sumSpeakerStats();
		Vy.centralizeStatistics(ubm.superVector(false, true, false));

		trainEigen(init, Vy);
	}

	public void trainEigenchannelMatrix(String init) throws IOException {

		if (Vy.eigen == null) {
			throw new IOException(
					"You have to train an EigenVoice Matrix first");
		}

		Ux.factors = new double[Ux.zero_stats.length][this.ec_iterations];
		Ux.centralizeStatistics(ubm.superVector(false, true, false));
		// removeSpeakerStatistics();
		trainEigen(init, Ux);
	}
	
	public void trainEigen(String init, JFA_Element elem) throws IOException {
		PrintWriter pw;
		
		if (init == null) {
			if (verbose)
				System.err.println("creating random " + elem.name + " matrix");
			elem.randomMatrix();
		} else {
			elem.loadEigenMatrix(init);
		}

		pw = new PrintWriter(elem.name + ".initial");

		if (verbose) {
			System.err.print("trainEigen (" + elem.name + ") "
					+ elem.eigen.getRowDimension());
			System.err.println(" ggg " + elem.eigen.getColumnDimension());
		}

		elem.eigen.transpose().print(pw, 1, 6);
		pw.close();

		for (int i = 0; i < elem.iterations; i++) {
			System.err.println("iteration: " + i);
			precomputeIndependent(elem);

			Matrix[] LU = estimate_LU_and_latent(elem, true);

			Matrix RU = estimateRU(elem, new Matrix(elem.factors));
			if (verbose) {
				System.err.println("RU row: " + RU.getRowDimension()
						+ " RU col: " + RU.getColumnDimension());
			}

			for (int g = 0; g < elem.nd; g++) {

				// System.err.println("LU row: " + LU[g].getRowDimension()
				// + " LU col: " + LU[g].getColumnDimension());

				// LU[g].inverse().print(1,3);
				for (int x = 0; x < elem.fd; x++) {
					int line = g * elem.fd + x;

					Matrix temp = RU.getMatrix(line, line, 0, elem.rank - 1);

					Matrix t2 = LU[g].solve(temp.transpose());

					elem.eigen.setMatrix(line, line, 0, elem.rank - 1, t2
							.transpose());
				}

			}

			System.out.println("new " + elem.name + " matrix created");
			pw = new PrintWriter(elem.name + ".it" + i);
			elem.eigen.transpose().print(pw, 1, 6);
			pw.close();
			if (verbose) {
				System.out.print(elem.eigen + " rows b: "
						+ elem.eigen.getRowDimension());
				System.out.println("  columns "
						+ elem.eigen.getColumnDimension() + " | "
						+ elem.eigen.rank());
			}
		}
	}

	

	/**
	 * calculate E[g]_t * iCova[g]
	 * 
	 * @param E
	 * @throws IOException
	 */
	public void precomputeIndependent(JFA_Element elem) throws IOException {

		if (!ubm.diagonal) {
			throw new IOException(
					"precomputeIndependent: UBM has to have diagonal cova matrix");
		}

		Matrix iCova = new Matrix(ubm.fd, ubm.fd, 0);

		// Precompute ET * iCova * E
		for (int g = 0; g < ubm.nd; g++) { // for each density
			// select proper part of Matrix E
			Matrix eigen_g = elem.eigen.getMatrix(g * ubm.fd,
					((g + 1) * ubm.fd) - 1, 0,
					elem.eigen.getColumnDimension() - 1);

			// set values of iCova Matrix
			for (int x = 0; x < ubm.fd; x++) {
				iCova.set(x, x, 1 / ubm.components[g].cov[x]);
				iSupervector[g * ubm.fd + x] = (1 / ubm.components[g].cov[x]);
			}

			Matrix iCova_E = iCova.times(eigen_g); // diemsion is ubm.fd * rank
			// Multiply EgT with iC_E.
			independent[g] = eigen_g.transpose().times(iCova_E);
			// contains now Eg_t *iCova[g]_E[g]
		}
	}

	/**
	 * remove statistics for eigenchannel computation; assumes that mean has
	 * already been removed formula: Ux.first_stats = Ux.first_stats -
	 * Ux.zero_stats * (V*y)
	 */
	public void removeSpeakerStatistics() {
		if (verbose) {
			System.err.println("Removing Speaker Statistics");
		}

		for (int s = 0; s < Vy.num; s++) {
			// For each speaker calculate V*y and substitute this vector from
			// all available sessions of this speaker

			double[] shift = Vy.eigen.times(new Matrix(Vy.factors[s], Vy.rank))
					.getRowPackedCopy();

			int[] mySessions = ht.get(speaker_idx.get(s));
			for (int i = 0; i < mySessions.length; i++) {

				for (int g = 0; g < Ux.nd; g++) {
					for (int x = 0; x < Ux.fd; x++) {
						Ux.first_stats[i][g * Ux.fd + x] -= Ux.zero_stats[i][g]
								* shift[g * Ux.fd + x];
					}
				}
			}
		}
	}

	/**
	 * remove statistics for eigenchannel computation; assumes that mean has
	 * already been removed formula: Vy.first_stats = Vy.first_stats -
	 * Vy.vero_stats * (U*x)
	 */
	public void removeChannelStatistics() {
		if (verbose) {
			System.err.println("Removing Channel Statistics");
		}
	}

	/**
	 * calculate L(s) = I + N(s) * eT * Sigma-1 * e for each density
	 * 
	 * @param E
	 *            the EigenXXX-Matrix, can be Eigenchannel or Eigenvoice
	 * @param zero_stats
	 *            zero statistics, in case of Eigenchannel, these are the
	 *            statistics for each session in case of Eigenvoice, these are
	 *            the summed statistics for each speaker
	 * @throws IOException
	 * @throws IOException
	 */
	public Matrix[] estimate_LU_and_latent(JFA_Element elem, boolean reestimate)
			throws IOException {

		if (independent == null) {
			throw new IOException(
					"compute_L_matrices: need precomputed matrices");
		}

		Matrix[] LU = new Matrix[ubm.nd];
		for (int i = 0; i < ubm.nd; i++) {
			LU[i] = new Matrix(elem.rank, elem.rank, 0);
		}

		Matrix ident = Matrix.identity(elem.rank, elem.rank);

		if (!ubm.diagonal) {
			throw new IOException(
					"compute_L_matrices: UBM has to have diagonal cova matrix");
		}

		// for each sessions or speakers
		for (int s = 0; s < elem.zero_stats.length; s++) {

			// System.err.println("LU calculation for session: "+this.session_idx.get(s));

			Matrix L_s = new Matrix(elem.rank, elem.rank, 0);
			for (int g = 0; g < ubm.nd; g++) {
				L_s.plusEquals(independent[g].times(elem.zero_stats[s][g]));
			}
			L_s.plusEquals(ident);

			// now solve according channel or speaker factors for the specific
			// session or speaker

			// returns a vector of size ubm.nd * ubm.fd
			double[] iSV_temp = iSupervector.clone();
			componentWiseMult(iSV_temp, elem.first_stats[s]);

			Matrix b = (elem.eigen.transpose()).times(new Matrix(iSV_temp,
					iSV_temp.length));

			// System.err.println("Dimension of Matrix b: "+ b.getRowDimension()
			// + " " +b.getColumnDimension());

			Matrix factor = L_s.solve(b);

			if (verbose) {
				// System.out.println("Rank of factors: " + factor.rank());
				// System.out.println("Row and column dimension of factors: "
				// + factor.getRowDimension() + " "
				// + factor.getColumnDimension());
				// System.out.println("factors:");
				// factor.transpose().print(1, 4);
			}
			/*
			 * if (verbose) { factor.transpose().print(1,7); }
			 */
			elem.factors[s] = factor.getRowPackedCopy();

			if (reestimate) {

				Matrix Linv = L_s.inverse();
				// System.out.println("Linv");
				// Linv.print(1, 6);
				Matrix temp = factor.times(factor.transpose());

				Linv.plusEquals(temp);

				for (int g = 0; g < ubm.nd; g++) {
					LU[g].plusEquals(Linv.times(elem.zero_stats[s][g]));
				}
			}
		}
		return LU;
	}

	public Matrix estimateRU(JFA_Element elem, Matrix factors) {

		// Matrix RU = (factors.transpose()).times(new
		// Matrix(elem.first_stats));
		Matrix RU = new Matrix(elem.fd * elem.nd, elem.rank);

		for (int s = 0; s < elem.first_stats.length; s++) {
			for (int g = 0; g < elem.nd; g++) {

				for (int x = 0; x < elem.fd; x++) {

					for (int r = 0; r < elem.rank; r++) {
						RU.set(g * elem.fd + x, r, elem.first_stats[s][g
								* elem.fd + x]
								* factors.get(s, r));
					}
				}
			}
		}
		// RU.print(1,2);
		return RU;
	}

	/**
	 * Calculate vec * vec^T and save it in Matrix Object
	 * 
	 * @param vec
	 * @return
	 */
	public Matrix vecTimesvecT(double[] vec) {
		Matrix ret = new Matrix(vec.length, vec.length);

		for (int i = 0; i < vec.length; i++) {
			for (int j = 0; j < vec.length; j++) {
				ret.set(i, j, vec[i] * vec[j]);
			}
		}
		return ret;
	}

	public void componentWiseMult(double[] vec1, double[] vec2)
			throws IOException {

		if (vec1.length != vec2.length) {
			throw new IOException(
					"Vector length of the two vectors is different");
		}
		for (int x = 0; x < vec1.length; x++) {
			vec1[x] = vec1[x] * vec2[x];
		}
	}

	public static void main(String[] args) {

		if (args.length != 3) {
			System.err.println("Wrong number of arguments");
			System.exit(-1);
		}
		// String ubmFile = args[0];
		String mean = args[0];
		String var = args[1];
		String tableFile = args[2];

		JFA_Trainer jfa = null;
		try {
			// jfa = new JFA_Trainer(ubmFile, tableFile);
			jfa = new JFA_Trainer(mean, var, tableFile);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (ClassNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		jfa.getMemInfo();
		try {
			jfa.loadOrComputeStatistics();
			System.out.println(jfa.Ux.first_stats.length);
			System.out.println(jfa.Ux.first_stats[0].length);
			jfa.trainEigenvoiceMatrix("ev.init");
			jfa.trainEigenchannelMatrix("ec.init");

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		jfa.getMemInfo();

		/*
		 * System.err.println("Zero statistics");
		 * 
		 * System.out.println(jfa.zero_stats.length + " " +
		 * jfa.zero_stats[0].length);
		 * 
		 * 
		 * for(int x=0;x<jfa.zero_stats.length;x++){
		 * 
		 * for(int y=0;y<jfa.zero_stats[x].length;y++){
		 * 
		 * System.out.print(jfa.zero_stats[x][y]+" "); } System.out.println();
		 * 
		 * }
		 */
	}
}
