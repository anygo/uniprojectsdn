package statistics.jfa;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import Jama.Matrix;

/**
 * JFA_Element contains all important FactorAnalysis components (factors,
 * matrix, statistics) for each JFA-Element
 * 
 * 
 * @author bocklet
 * 
 */
public class JFA_Element {

	private boolean verbose = true;

	public boolean diagonal;
	public String name = "";
	public Matrix eigen; // eigenmatrix (Eigenchannel or Eigenvoice
	public double[][] zero_stats; // zero order statistics
	public double[][] first_stats; // first order statistics
	public double[][] factors;
	public int iterations = 0, // number of iterations for training ,
			rank = 0, // rank of eigen matrix
			num, // number of sessions
			nd, fd; // number of densities and feature dimension

	/**
	 * This constructor is used to for training the eigen matrix. zero_stats and
	 * first_stats are initialized properly
	 * 
	 * @param session
	 * @param nd
	 * @param fd
	 * @param rank
	 * @param iterations
	 */
	public JFA_Element(int num, int nd, int fd, int rank, int iterations,
			String name, boolean diagonal) {

		this.eigen = null;
		this.iterations = iterations;
		this.rank = rank;
		this.nd = nd;
		this.fd = fd;
		this.num = num;
		this.zero_stats = new double[num][nd];
		this.first_stats = new double[num][nd * fd];
		this.name = name;
		this.diagonal = diagonal;
	}

	/**
	 * This constructor is used to read the corresponding eigenmatrix from file
	 * 
	 * @param matrixFile
	 *            matrix file
	 * @param iterations
	 *            number of iterations for training
	 * @throws IOException
	 */
	public JFA_Element(String matrixFile, int iterations, int fd, int nd,
			boolean diagonal) throws IOException {

		loadEigenMatrix(matrixFile);
		this.iterations = iterations;
		this.fd = fd;
		this.nd = nd;
		this.diagonal = diagonal;
	}

	public void loadEigenMatrix(String file) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		eigen = Matrix.read(br).transpose();
		this.rank = eigen.getRowDimension(); // NOT SURE IF THIS IS CORRECT,
		// HAVE TO CHECK IT!!!!!
		if (verbose) {
			System.err.print("EigenMatrix " + file + " read in");
			System.err.println(" eith dimension (rowDim x colDim): "
					+ eigen.getRowDimension() + " "
					+ eigen.getColumnDimension());
		}
		br.close();
	}

	public void randomMatrix() {
		eigen = Matrix.random(nd * fd, rank);
	}

	public void centralizeStatistics(double[] shift) {

		if (verbose) {
			System.err
					.println("Centralizing first (and second-order) Statistics");
		}

		// first order statistics

		for (int s = 0; s < num; s++) {

			if (verbose) {
				System.err.println();
				System.err.println(first_stats[s][0] + " " + first_stats[s][1]
						+ " " + first_stats[s][2]);
			}

			// first_stats_d = first_stats_d - zero_stats_d * m_d
			for (int d = 0; d < nd; d++) {

				for (int i = 0; i < fd; i++) {
					first_stats[s][d * fd + i] -= (zero_stats[s][d] * shift[d
							* fd + i]);
				}
			}
			if (verbose) {
				System.err.println("after");
				System.err.println(first_stats[s][0] + " " + first_stats[s][1]
						+ " " + first_stats[s][2]);
			}
		}
		if (verbose) {
			System.err.println("END centrealzie Statistics");
		}
	}

}
