import java.io.*;

public class Timetest {

public static void main(String[] args) {
for (int i = 1; i < 1001; i += 10) {
		int[] check = new int[i];

for(int j = 0; j < check.length; j++) {
			check[j] = j;
		}
	}
	Laufzeit.init();
}}
