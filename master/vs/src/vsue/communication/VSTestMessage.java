package vsue.communication;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;

import java.io.ObjectOutput;

public class VSTestMessage implements Externalizable {

	private static final long serialVersionUID = 1L;
	
	private int integer;
	private String string;
	private Object[] objects;
	
	public VSTestMessage() {}
	
	public VSTestMessage(int integer, String string, Object[] objects) {
		this.integer = integer;
		this.string = string;
		this.objects = objects;
	}

	private void writeByteFromInt(int i, ObjectOutput out) throws IOException {
		out.write(i);
		out.write(i >> 8);
		out.write(i >> 16);
		out.write(i >> 24);
		
	}
	
	private void writeByteFromShort(int i, ObjectOutput out) throws IOException {
		out.write(i);
		out.write(i >> 8);
	}
	
	private int readIntFromByte(ObjectInput in) throws IOException {
		
		int ret;
		
		int a = in.read();
		int b = in.read();
		int c = in.read();
		int d = in.read();
		
		ret = d << 8;
		ret = ret | c;
		ret = ret << 8;
		ret = ret | b;
		ret = ret << 8;
		ret = ret | a;
		
		return ret;
	}
	
	private short readShortFromByte(ObjectInput in) throws IOException {
		
		short ret;
		
		short a = (short) in.read();
		short b = (short) in.read();
		
		ret = (short) (b << 8);
		ret = (short) (ret | a);
		
		return ret;
	}
	
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		//System.out.println("writeExternal");
		
		// integer
		writeByteFromInt(integer, out);
		
		// string
		writeByteFromInt(string.length(), out);
		for (int i = 0; i < string.length(); ++i) {
			out.writeChar(string.charAt(i));
		}
		
		// object[]
		writeByteFromShort(objects.length, out);
		for (int i = 0; i < objects.length; ++i) {
			out.writeObject(objects[i]);
		}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		//System.out.println("readExternal");
		
		// integer
		this.integer = readIntFromByte(in);
		
		// string
		string = "";
		int size = readIntFromByte(in);
		for (int i = 0; i < size; ++i) {
			string += in.readChar();
		}
		
		// object[]
		short sizeArray = readShortFromByte(in);
		objects = new Object[sizeArray];
		for (short i = 0; i < sizeArray; ++i) {
			this.objects[i] = in.readObject();
		}
	}
	
	
	

}
