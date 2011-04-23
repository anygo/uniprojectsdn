package util;

import statistics.MixtureDensity;

public class Gnuplot {
	public static final String SYNOPSIS = 
		"usage: statistics.MixtureDensity data-file <details:none|id|details> mixture1 [mixture2 ...]";
	
	public static void main(String [] args) throws Exception {
		if (args.length < 3) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		int desc = 0;
		if (args[1].equals("none"))
			desc = 0;
		else if (args[1].equals("id"))
			desc = 1;
		else if (args[1].equals("details"))
			desc = 2;
		else
			throw new Exception("illegal parameter " + args[1]);
		
		System.out.println("set term png");
		System.out.println("set parametric");
		System.out.println("plot '" + args[0] + "' w d notitle, \\");
		
		for (int i = 2; i < args.length; ++i) {
			MixtureDensity md = MixtureDensity.readFromFile(args[i]);
			for (int j = 0; j < md.nd; ++j) {
				int oldid = md.components[j].id;
				md.components[j].id = j;
				System.out.print(md.components[j].covarianceAsGnuplot());
				if (desc == 0)
					System.out.print(" notitle");
				else if (desc == 1)
					System.out.print(" t '" + j + "'");
				else if (desc == 2)
					System.out.print(String.format("t '%.2f,%.2f : %.2f'", md.components[j].mue[0], md.components[j].mue[1], md.components[j].apr));
				else
					throw new Exception("illegal desc type " + desc);
				
				if (j < md.nd - 1) 
					System.out.println(", \\");
				md.components[j].id = oldid;
			}
		}
	
		System.out.println(";");
		
		System.out.println("quit");
	}
}
