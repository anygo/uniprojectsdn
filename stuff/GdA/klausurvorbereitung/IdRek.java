//AAAHH wieso funktioniert der dreck nicht?  - eha.. war ja nicht sortiert.. ololo
public class IdRek {
	
	public static boolean hatIdR(int[] arr, int von, int bis) {
		if (von < 0 || bis < 0) return false;
		if (von == bis) {
			if (arr[bis] == bis) return true;
			return false;
		}
		int mid = (von+bis)/2;
		if (arr[mid] == mid) return true;
		if (arr[mid] > mid) return hatIdR(arr, von, mid-1);
		return hatIdR(arr, mid+1, bis);
	}

	public static void main(String[] args) {
		int[] arr = new int[6];
		arr[0] = 0;
		arr[1] = 2;
		arr[2] = 4;
		arr[3] = 5;
		arr[4] = 44;
		arr[5] = 100;

		System.out.println(hatIdR(arr, 0, 5));
	}
}
