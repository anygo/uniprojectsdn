package vsue.marshalling;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.nio.*;

public class VSExternalizableExample implements VSExternalizable {

	private int a;
	private double b;
	private char[] c;

	public VSExternalizableExample(int a, double b, char[] c) {

		this.a = a;
		this.b = b;
		this.c = c;

	}

	public String toString() {

		return new String("a = "+a+" ; b = "+b+" ; c = \""+ new String(c) +"\"");

	}

	public void writeExternal(VSObjectOutputStream out) throws IOException {

		OutputStream os = out.getOutputStream();
		int size = 2 * Integer.SIZE/8 + Double.SIZE/8 + c.length * Character.SIZE/8;
		ByteBuffer buf = ByteBuffer.allocate(size);
		buf.putInt(size);
		buf.putInt(a);
		buf.putDouble(b);
		for(int i = 0; i < c.length; i++) {
			buf.putChar(c[i]);
		}
		os.write(buf.array());

	}

	public void readExternal(VSObjectInputStream in) throws IOException {

		InputStream is = in.getInputStream();
		ByteBuffer buf;

		buf = getByteBuffer(is, Integer.SIZE/8);
		int size = buf.getInt();
		
		buf = getByteBuffer(is, size - Integer.SIZE/8);
		a = buf.getInt();
		b = buf.getDouble();
		int charArrayLen = (size - ( 2 * Integer.SIZE/8 + Double.SIZE/8)) / (Character.SIZE/8);
		c = new char[charArrayLen];
		for(int i = 0; i < c.length; i++) {
			c[i] = buf.getChar();
		}

	}

	private ByteBuffer getByteBuffer(InputStream in, int size) throws IOException {
		int start, res;
		int bytepos = 0;
		byte[] byteArr = new byte[size];
		while(bytepos < size) {
			res = in.read(byteArr, bytepos, size - bytepos);
			if(res > 0) {
				bytepos += res;
			} else {
				throw new IOException();
			}
		}
		return ByteBuffer.wrap(byteArr);
	}
}
	
