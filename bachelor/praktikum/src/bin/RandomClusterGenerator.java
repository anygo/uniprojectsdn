package bin;

import java.io.IOException;

import statistics.DensityDiagonal;
import io.FrameWriter;

public class RandomClusterGenerator {

	public static final String SYNOPSIS = 
		"usage: java RandomClusterGeneratr density-string1 number1 [density-string2 number2...] > feature-file\n" +
		"\n" +
		"density-string: mue1,mue2,cov1,cov2";
	
	public static void main(String[] args) throws IOException {
		if (args.length < 2 || args.length % 2 != 0) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		int fd = 2;
		
		FrameWriter fw = new FrameWriter(fd);
		
		for (int i = 0; i < args.length; ++i) {
			DensityDiagonal d = DensityDiagonal.fromString(args[i++]);
			int ns = Integer.parseInt(args[i]);
			for (int j = 0; j < ns; ++j)
				fw.write(d.drawSample());
		}
		
		fw.close();
	}

}
