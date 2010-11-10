
public class BinarySearch {
	
	static boolean binarySearchRec(int[] x, int value) {
		return (binS(x, value, 0, x.length-1));
	}

	static boolean binS(int[] x, int value, int firsti, int lasti) {
		int tmpi = (firsti + lasti)/2;
		if (x[tmpi] == value) return true;
		if (firsti >= lasti && x[firsti] != value) return false;
		if (x[tmpi] > value) return binS(x, value, firsti, tmpi-1);
		else return binS(x, value, tmpi+1, lasti);
	}

	static boolean binarySearchNonRec(int[] x, int value) {
		int tmpi = (x.length-1)/2;
		int firsti = 0;
		int lasti = x.length -1;
		while (firsti < lasti) {
			if (x[tmpi] == value) return true;
			if (x[tmpi] > value) {
				lasti = tmpi-1;
				tmpi = (firsti+lasti)/2;
			}
			else {
				firsti = tmpi+1;
				tmpi = (firsti+lasti)/2;
			}
		}
		if (x[firsti] == value) return true;
		return false;
	}

	public static void main(String[] args) {
	int[] x = new int[100];
	int key = Integer.parseInt(args[0]);
	for (int i = 0; i < 100; i++) x[i] = i*i;
	
	System.out.println("Rec: "+binarySearchRec(x, key)+"; NonRec: "+binarySearchNonRec(x, key));
	}
}
