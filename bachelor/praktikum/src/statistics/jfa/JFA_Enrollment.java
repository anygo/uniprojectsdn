package statistics.jfa;

import java.io.IOException;

import statistics.MixtureDensity;

/**
 * JFA_Enrollment calculates latent of a given utterance in the enrollment the
 * speaker models are trained and saved to a specified directory It is
 * important, that the matrices U, V and D have been trained already
 * 
 * 
 * @author bocklet
 * 
 */
public class JFA_Enrollment extends JFA {

	public JFA_Enrollment(MixtureDensity ubm, String UMatrix, String VMatrix,
			String DMatrix, int iterations, boolean features) {

		super(ubm,features);
		try {
			this.Vy = new JFA_Element(VMatrix, iterations, ubm.fd, ubm.nd, false);
		} catch (IOException e) {
			System.err.println("EigenvoiceMatrix file " + VMatrix
					+ " not available.");
			System.err.println("--- running Training without EigenChannel");
			this.Vy = null;
		}
		
		try {
			this.Ux = new JFA_Element(UMatrix, iterations, ubm.fd , ubm.nd, false);
		} catch (IOException e) {
			System.err.println("EigenvoiceMatrix file " + UMatrix
					+ " not available.");
			System.err.println("--- running Training without EigenChannel");
			this.Ux = null;
		}
		
		try {
			this.Dz = new JFA_Element(DMatrix, iterations, ubm.fd , ubm.nd, false);
		} catch (IOException e) {
			System.err.println("EigenvoiceMatrix file " + UMatrix
					+ " not available.");
			System.err.println("--- running Training without EigenChannel");
			this.Dz = null;
		}
	}
	
	
	

}
