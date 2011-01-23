import java.io.*;

public class Programm {
	
	public static void main(String[] args) {
		Laufzeit.init();
		try {
			File file = new File("function3.txt");
			BufferedWriter output = new BufferedWriter(new FileWriter("function3.txt"));

			for (int i = 1; i < 2500; i += 10) {
				int[] tmp = new int[i];
				for (int j = 0; j < i; j++) {
					tmp[j] = j;
				}
				long startTime = System.nanoTime();
				Laufzeit.function3(tmp);
				long endTime = System.nanoTime();
				long time = endTime - startTime;
					
				String text = i + " " + time;
				output.write(text);
				output.newLine();
			}
			output.close();
		} catch (Exception e) {
			System.out.println("Error");
		}
	}
}
