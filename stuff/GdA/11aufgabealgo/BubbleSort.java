
public class BubbleSort {
	
	public static void sort(int[] array) {
		
		boolean swapped;
		int upper = array.length - 1;

		do {
			swapped = false; 
			for (int i = 0; i < upper; i++) {
				if (((Comparable) array[i]).compareTo(array[i+1]) > 0) {
					swap(array, i, i+1);
					swapped = true;
				}
			}
			upper--;

		} while (swapped);
	}

	public static void swap(int[] array, int index_1, int index_2) {
		
		int tmp = array[index_1];
		array[index_1] = array[index_2];
		array[index_2] = tmp;
	}
	
	public static void print(int[] arr) {
		
		System.out.print("SortedArray:");
		for (int i = 0; i < arr.length; i++) {
			System.out.print(" "  + arr[i]);
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
		long a = System.currentTimeMillis();
		sort(arr);
		long b = System.currentTimeMillis();
		System.out.println(b-a);
	}
}
