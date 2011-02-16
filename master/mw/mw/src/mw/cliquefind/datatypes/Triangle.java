package mw.cliquefind.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class Triangle implements org.apache.hadoop.io.WritableComparable<Triangle> {

	public int a, b, c;
	
	public Triangle() {
	}
	
	public Triangle(int a, int b, int c) {
		this.a = a;
		this.b = b;
		this.c = c;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		a = in.readInt();
		b = in.readInt();
		c = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(a);
		out.writeInt(b);
		out.writeInt(c);
	}

	@Override
	public int compareTo(Triangle o) {
		if (a == o.a && b == o.b && c == o.c )
			return 0;
		else if (a != o.a)
			return a - o.a;
		else if (b != o.b)
			return b - o.b;
		else 
			return c - o.c;
	}
	
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Triangle)) return false;
		Triangle z = (Triangle)o;
		return (a == z.a && b == z.b && c == z.c);
	}
	
	@Override
	public int hashCode() {
		return (a * 2342) ^ (b * 4711) ^ (b * 2213);
	}

}
