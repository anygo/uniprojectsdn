package mw.cliquefind.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;


public class WeightedEdge implements WritableComparable<WeightedEdge> {

	public int a, b;
	public long weight;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		a = in.readInt();
		b = in.readInt();
		weight = in.readLong();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(a);
		out.writeInt(b);
		out.writeLong(weight);
		
	}

	@Override
	public int compareTo(WeightedEdge o) {
		int min = Math.min(a,b);
		int max = Math.max(a,b);
		long w  = weight;
		
		int minOther = Math.min(o.a, o.b);
		int maxOther = Math.min(o.a, o.b);
		long wOther  = o.weight;
		
		if (min == minOther && max == maxOther && w == wOther) 
			return 0;
		else if ((min < minOther) || (min == minOther && max < maxOther) || (min == minOther && max == maxOther && w < wOther))
			return -1;
		else
			return 1;
	}
	
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof WeightedEdge)) return false;
		WeightedEdge z = (WeightedEdge)o;
		return (a == z.a && b == z.b && weight == z.weight);
	}
	
	@Override
	public int hashCode() {
		return (int) ((a * 2342) ^ (b * 4711) ^ (weight * 44321));
	}

}
