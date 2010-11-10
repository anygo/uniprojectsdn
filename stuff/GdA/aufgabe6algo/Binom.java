
public class Binom {

	public static long fak_rek(long n) {
		if (n == 1 || n == 0)
		{
			return 1;
		} 
		if (n < 0)	
		{
			System.out.println("### Kann die Fakultaet einer negativen Zahl nicht berechnen!");
			return -1;
		}
		return n * fak_rek(n-1);
	}
	
	public static long binom_rek(long n, long k) {
		if (k == 0 || k == n) 
		{
			return 1;
		}
		return (binom_rek(n-1, k-1) + binom_rek(n-1, k));
	}
	
	public static long binom_closed(long n, long k) {
		return (fak_rek(n) / (fak_rek(k) * fak_rek(n-k))); 
	}
	
	public static long binom_closed_opt(long n, long k) {
		if (k < n-k)
		{
			k = n - k;
		}
		long zaehler = 1;
		long nenner = fak_rek(n-k);
		for (long i = n; i > k; i--)
		{
			zaehler *= i;
		}
		return (zaehler / nenner);
	}

	public static void main(String[] args) {
	
		long n = Long.parseLong(args[0]);
		
		System.out.println("\n\nBinom_closed_opt");
		
		for (long i1 = 0; i1 <= n; i1++) {
			System.out.print(binom_closed_opt(n,i1) + " ");
		}	
		
		System.out.println("\n\nBinom_rek");

		for (long i1 = 0; i1 <= n; i1++) {
			System.out.print(binom_rek(n,i1) + " ");
		}

		System.out.println("\n\nBinom_closed");
		
		for (long i1 = 0; i1 <= n; i1++) {
			System.out.print(binom_closed(n,i1) + " ");
		}
	}
}
