	
public class BinarySearch {	

	public static int binarySearch(int[] sortedArray, int key) {
	
		int left = 0;
		int right = sortedArray.length - 1;
		int mid = (right + left)/2;

		while (left <= right - 1) {

			if (sortedArray[mid] == key)
				return mid;
			if (sortedArray[mid] > key) {
				right = mid - 1;
				mid = (right + left)/2;
			} else {
				left = mid + 1;
				mid = (right + left)/2;
			}
		}

		return -1;
	}
	
	public static void main(String[] args) {
		
		if (args.length < 1) {
			System.err.println("Programmaufruf mittels java BinarySearch [KEY(s)]");
			return;
		}

		int[] querys = new int[args.length];
	
		for (int i = 0; i < args.length; i++) {
			try {
				querys[i] = Integer.parseInt(args[i]);
			} catch (Exception e) {
				System.err.println("Als Parameter nur positive Integerwerte - Abbruch.");
				return;
			}
		}
	

		int[] arr = new int[2000];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (int) (java.lang.Math.random() * 1000);
		}
		MergeSort.sort(arr);

		for (int i = 0; i < querys.length; i++) {
			if (binarySearch(arr, querys[i]) != -1) {
				System.out.println(querys[i] + " ist enthalten");
			} else {
				System.out.println(querys[i] + " ist nicht enthalten");
			}
		}
	}
}
