
public class MergeSort {
	
	public static void sort(int[] x) {
		
		if (x.length <= 1) 
			return;

		int[] x1 = new int[x.length/2];
		int[] x2 = new int[x.length - x.length/2];
		
		for (int j = 0; j < x.length/2; j++) {
			x1[j] = x[j];
		}
		for (int j = 0; j < x.length - x.length/2; j++) {
			x2[j] = x[j + x.length/2];
		}

		sort(x1);
		sort(x2);

		int x1_ = 0;	// "x1_pointer"
		int x2_ = 0;	// "x2_pointer"
		int i = 0;

		while (x1_ < x1.length && x2_ < x2.length) {
			if (x1[x1_] <= x2[x2_]) {
				x[i++] = x1[x1_++];
			} else {
				x[i++] = x2[x2_++];
			}
		}
		while (x1_ >= x1.length && i < x.length)
			x[i++] = x2[x2_++];
		
		while (x2_ >= x2.length && i < x.length)
			x[i++] = x1[x1_++];

	}

	public static void print(int[] arr) {
		
		for (int i = 0; i < arr.length; i++) {
			System.out.print(arr[i]+" ");
		}
		System.out.println();
	}

	public static void main(String[] args) {
		
		if (args.length != 1) {
			System.out.println("Programmaufruf mittels java BubbleSort ARRAYLENGTH");
			return;
		}

		int size = 0;
		
		try {
			size = Integer.parseInt(args[0]);
		} catch (Exception e) {
			System.out.println("Als Parameter nur positive Integerwerte - Abbruch.");
			return;
		}
		
		if (size <= 0) {
			System.out.println("Als Parameter nur positive Integerwerte - Abbruch.");
			return;
		}
		
		int[] arr = new int[size];
		for (int i = 0; i < size; i++) {
			arr[i] = (int) (java.lang.Math.random() * 1000);
		}
		
		System.out.print("Zufaelliges Array: ");
		print(arr);
		System.out.println("\nSortiere...");
		long a = System.nanoTime();
		sort(arr);
		long b = System.nanoTime();;
		long time = b - a;
		System.out.print("\nSortieren abgeschlossen nach " + time + "ns\n\nSortiertes Array: ");
		print(arr);
	}
}
