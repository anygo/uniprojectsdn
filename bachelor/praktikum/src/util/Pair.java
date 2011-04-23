package util;

/**
 * Use the Pair class to group two values together.
 * 
 * @author sikoried
 */
public class Pair<T1, T2> {
	public Pair(T1 a, T2 b) {
		this.a = a;
		this.b = b;
	}
	public T1 a;
	public T2 b;
	public boolean equals(Object p) {
		if (!(p instanceof Pair<?, ?>)) 
			return false;
		Pair <?, ?> pair = (Pair<?, ?>) p;
		return pair.a.equals(a) && pair.b.equals(b);
	}
}
