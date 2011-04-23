package agreement;

import statistics.*;

/**
 * Compute mue and sigma of a given random variable (double array).
 * 
 * @author sikoried
 */
public class MueSigma {
	public static String synopsis = 
		"usage: MueSigma data-file-1 <data-file-2 ...>";
	
	public static void main(String[] args) throws Exception {
		for (int i = 0; i < args.length; ++i) {
			try {
				DataSet ds = new DataSet(args[i]);
				ds.fromAsciiFile(args[i], -1);
				Density d = Trainer.ml(ds.samples, true);
				System.out.print(args[i] + ": mue = [");
				for (double dd : d.mue)
					System.out.print(" " + dd);
				System.out.print(" ] sig = [");
				for (double dd : d.cov)
					System.out.print(" " + Math.sqrt(dd));
				System.out.println(" ]");
			} catch (Exception e) {
				System.out.println(e);
				e.printStackTrace();
			}
		}
	}
}
