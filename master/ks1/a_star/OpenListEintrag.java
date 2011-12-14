package astar;

public class OpenListEintrag implements Comparable<OpenListEintrag> {

	public NPuzzle destination;
	public OpenListEintrag precessor;
	public int g;

	public OpenListEintrag(NPuzzle destination, OpenListEintrag precessor, int g) {
		super();
		this.destination = destination;
		this.precessor = precessor;
		this.g = g;
	}

	@Override
	public int compareTo(OpenListEintrag other) {

		int ha = this.g + Heuristics.h1(this.destination);
		int hb = other.g + Heuristics.h1(other.destination);

		if (ha < hb) {
			return -1;
		}
		if (ha > hb) {
			return 1;
		}
		return 0;
	}
}
