
public class Problem100 {
	public static void algo(int n) {
		System.out.print(n + " ");

		if (n == 1) return;
		if (n % 2 != 0) {
			algo(3*n+1);
			return;
		} else {
			algo(n/2);
			return;
		}
	}

	public static void main(String[] args) {
		int n = Integer.parseInt(args[0]);
		algo(n);
	}
}
