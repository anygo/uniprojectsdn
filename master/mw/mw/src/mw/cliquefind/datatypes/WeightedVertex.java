package mw.cliquefind.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class WeightedVertex implements org.apache.hadoop.io.WritableComparable<WeightedVertex> {
	
	public int id;
	public long weight;	
	
	@Override
	public void readFields(DataInput in) throws IOException {
		id = in.readInt();
		weight = in.readLong();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(id);
		out.writeLong(weight);
	}

	@Override
	public int compareTo(WeightedVertex o) {
		if( weight > o.weight )
			return 1;
		else if( weight == o.weight ) 
			return 0;
		else
			return -1;
	}
	
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof WeightedVertex)) return false;
		WeightedVertex z = (WeightedVertex)o;
		return (id == z.id && weight == z.weight);
	}
	
	@Override
	public int hashCode() {
		return (int) ((id * 2342) ^ (weight * 4711));
	}

}
