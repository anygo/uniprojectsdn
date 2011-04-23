
public class Sort {
	
	public static void BubbleSort(int[] arr) {
		for (int end=arr.length; end > 0; end--) {
			for (int i = 1; i < end; i++) {
				if (arr[i-1] > arr[i]) {
					int tmp = arr[i];
					arr[i] = arr[i-1];
					arr[i-1] = tmp;
				}
			}
		}
	}
	
	public static void MergeSort(int[] arr) {
		if (arr.length <= 1) return;

		int[] arr1 = new int[(arr.length)/2];
		int[] arr2 = new int[(arr.length) - ((arr.length)/2)];

		for (int i = 0; i < arr1.length; i++) 
			arr1[i] = arr[i];

		for (int i = 0; i < arr.length - arr.length/2; i++)
			arr2[i] = arr[i + arr1.length];

		MergeSort(arr1);
		MergeSort(arr2);

		
		// MELTORRRRRR
		
		int arr1_p = 0;
		int arr2_p = 0;
		int i = 0;

		while (arr1_p < arr1.length && arr2_p < arr2.length) {
			if (arr1[arr1_p] <= arr2[arr2_p]) {
				arr[i++] = arr1[arr1_p++];
			} else {
				arr[i++] = arr2[arr2_p++];
			}

		while (arr1_p < arr1.length)
			arr[i++] = arr1[arr1_Ãp++
		}

	}

	public static void print(int[] arr) {
		for (int i = 0; i < arr.length; i++) 
			System.out.print(arr[i] + " ");
		System.out.println();
	}

	public static void main(String[] args) {
		int[] test = new int[15];
		int[] test2 = new int[15];
		for (int i = 0; i < test.length; i++) test[i] = (int) (java.lang.Math.random()*100);
		for (int i = 0; i < test2.length; i++) test2[i] = (int) (java.lang.Math.random()*100);
		

		print(test);
		BubbleSort(test);
		print(test);

		System.out.println("\n\n");
		print(test2);
		MergeSort(test2);
		print(test2);
	}
}
