package vsue.tests;

import java.io.Serializable;


/*
 * Message-Beispielklasse
 */
public class VSMessageExample implements Serializable {
	/*
	 * Testfelder verschiedenen Typs um Uebertragung zu testen
	 * (inklusive get-Methoden fuer private Variablen)
	 * */
	
	private static final long serialVersionUID = -6990470709599106224L;
	private short a;
	private int b;
	private long c;
	private boolean d;
	private float e;
	private double f;
	private char g;
	private byte h;
	public short i;
	public int j;
	public long k;
	public boolean l;
	public float m;
	public double n;
	public char o;
	public byte p;
	public char[] q;
	transient public short t1;
	transient public int t2;
	transient public long t3;
	transient public boolean t4;
	transient public float t5;
	transient public double t6;
	transient public char t7;
	transient public byte t8;
	transient public char[] t9;
	
	public VSMessageExample simple;
	public VSMessageExample[] sarray;
	
	public VSMessageExample() {
		
	}
	
	public VSMessageExample(short s, int i, long l, boolean bool, float f, double d, char c, byte b) {
		this.a = s;
		this.b = i;
		this.c = l;
		this.d = bool;
		this.e = f;
		this.f = d;
		this.g = c;
		this.h = b;
	}
	
	public short getShort () { return a; }
	public int getInt () { return b; }
	public long getLong () { return c; }
	public boolean getBoolean () { return d; }
	public float getFloat () { return e; }
	public double getDouble () { return f; }
	public char getChar () { return g; }
	public byte getByte () { return h; }
}
