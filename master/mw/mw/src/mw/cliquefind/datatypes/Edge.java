package mw.cliquefind.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;


public class Edge implements WritableComparable<Edge> {

	public int a, b;
	
	public Edge() {

	}
	
	public Edge(int a, int b) {
		this.a = a;
		this.b = b;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		a = in.readInt();
		b = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(a);
		out.writeInt(b);
	}

	@Override
    public int compareTo(Edge e) {

        int min = Math.min(a, b);
        int max = Math.max(a, b);
        
        int minOther = Math.min(e.a, e.b);
        int maxOther = Math.max(e.a, e.b);

        if (min == minOther && max == maxOther) 
            return 0;
        else if (min < minOther) 
            return -1; 
        else if (min == minOther && max < maxOther)
            return -1; 
        else
        	return 1;
    }  
	
    @Override
    public String toString() {
        return a + " -> " + b;
    } 
	
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Edge)) {
			System.out.println("KEINE EDGE! (equals funktion) -> false");
			return false;
		}
		return compareTo((Edge)o) == 0 ? true : false;
	}
	
	@Override
	public int hashCode() {
		return (a * 2342) ^ (b * 4711);
	}

}
