package util;

import statistics.MixtureDensity;

public class MixtureToAscii {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		for (String s : args) {
			MixtureDensity md = MixtureDensity.readFromFile(s);
			System.out.println(md);
		}
	}

}
