package vsue.tests;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.NotSerializableException;
import java.io.OutputStream;

import junit.framework.TestCase;
import vsue.marshalling.VSObjectInputStream;
import vsue.marshalling.VSObjectOutputStream;

//import com.sun.security.auth.module.UnixSystem;

public class VSMarshallingTest extends TestCase {
	public OutputStream stream = null;
	private InputStream inStream = null;
	private OutputStream outStream = null;
	private VSObjectInputStream in = null;
	private VSObjectOutputStream out = null;
	private int id = 0;
	public void init () {
		
		// Initialisierung
		try {
			int id = (int) (Math.random() * 10000);
			/* doesn't seem to work on many systems
			UnixSystem sys = new UnixSystem();
			int uid = (int) sys.getUid();
			*/
			outStream = new FileOutputStream("/tmp/temp" + id);
			inStream = new FileInputStream("/tmp/temp" + id);
		} catch (IOException e) {
			fail("init() failed: " + e);
		}
		in = new VSObjectInputStream(inStream);
		out = new VSObjectOutputStream(outStream);
	}
	
	public void end () {
		// Aufraeumen
		File tmp = new File("/tmp/temp" + id);
		tmp.delete();
		try {
			in.close();
			out.close();
			tmp = new File("/tmp/temp" + id);
			tmp.delete();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public VSMarshallingTest(String name) {
		super(name);
	}
	
	public void testNotSerializableException() {
		// Testen ob NotSerializableException richtig geworfen wird
		init();
		
		try {
			out.writeObject(new Object());
			
			// Keine Exception geworfen
			fail("NotSerializableException missig!");
		} catch (NotSerializableException e) {
			
			// Alles korrekt
			
		} catch (IOException e) {
			fail("unexpected IOException:" + e);
		}
		
		end();
	}
			
	public void testMessageExample() {
		// Ein VSMessageExample uebertragen
		init();
		
		// VSMessageExample erzeugen und initialisieren
		VSMessageExample so = new VSMessageExample((short) 15, 15000, Long.MAX_VALUE, true, (float)4.56, 1.234567, 'C', (byte) 200);
		so.o = 'Y';
		so.q = new char[10];
		for(int i = 0; i < 10; i++)
			so.q[i] = (char) ('A' + i);
		so.t1 = 5;
		so.t2 = 10;
		so.t3 = 15;
		so.t4 = (! so.t4);
		so.t5 = 5.5f;
		so.t6 = 1.11d;
		so.t7 = 'f';
		so.t8 = (byte) 25;
		so.t9 = (new String("Hallo")).toCharArray();
				
		VSMessageExample rcv = null;
		try {
			// Objekt schreiben und lesen
			out.writeObject(so);
			rcv = (VSMessageExample) in.readObject();
		} catch (Exception e) {
			e.printStackTrace();
			fail("Read/Write Object failed!");
		}
		
		// Uebertragung testen 
		assertTrue("char array was not transmitted correctly", (new String(rcv.q)).equals(new String(so.q)));
		assertTrue("char field was not transmitted correctly", rcv.o == 'Y');
		assertTrue("short field was not transmitted correctly", rcv.getShort() == (short) 15);
		assertTrue("int field was not transmitted correctly", rcv.getInt() == 15000);
		assertTrue("long field was not transmitted correctly", rcv.getLong() == Long.MAX_VALUE);
		assertTrue("boolean field was not transmitted correctly", rcv.getBoolean() == true);
		assertTrue("float field was not transmitted correctly", rcv.getFloat() == (float)4.56);
		assertTrue("double field was not transmitted correctly", rcv.getDouble() == 1.234567);
		assertTrue("char field was not transmitted correctly", rcv.getChar() == 'C');
		assertTrue("byte field was not transmitted correctly", rcv.getByte() == (byte) 200);
		
		// Testen, dass transient-Felder nicht uebertragen werden
		//System.out.println(so.t1 + " - " + rcv.t1);
		assertTrue("transient field should not be transmitted", so.t1 != rcv.t1);
		assertTrue("transient field should not be transmitted", so.t2 != rcv.t2);
		assertTrue("transient field should not be transmitted", so.t3 != rcv.t3);
		assertTrue("transient field should not be transmitted", so.t4 != rcv.t4);
		assertTrue("transient field should not be transmitted", so.t5 != rcv.t5);
		assertTrue("transient field should not be transmitted", so.t6 != rcv.t6);
		assertTrue("transient field should not be transmitted", so.t7 != rcv.t7);
		assertTrue("transient field should not be transmitted", so.t8 != rcv.t8);
		assertTrue("transient field should not be transmitted", so.t9 != rcv.t9);
		
		end();
	}
	
	public void testMessageExampleNested() {
		// Verschachteltes VSMessageExample verschicken
		init();
		
		// 3 VSMessageExamples anlegen und initialisieren
		VSMessageExample so = new VSMessageExample((short) 15, 15000, Long.MAX_VALUE, true, (float)4.56, 1.234567, 'C', (byte) 200);
		so.o = 'Y';
		so.q = new char[10];
		for(int i = 0; i < 10; i++)
			so.q[i] = (char) ('A' + i);
		
		VSMessageExample so2 = new VSMessageExample((short) 17, 17000, Long.MAX_VALUE - 3, false, (float)4.76, 1.234767, 'G', (byte) 70);
		so2.o = 'X';
		so2.q = new char[26];
		for(int i = 0; i < 26; i++)
			so2.q[i] = (char) ('A' + i);
		
		VSMessageExample so3 = new VSMessageExample((short) 17, 17000, Long.MAX_VALUE - 3, false, (float)4.76, 1.234767, 'G', (byte) 70);
		so2.o = 'X';
		so2.q = new char[26];
		for(int i = 0; i < 26; i++)
			so2.q[i] = (char) ('A' + i);
		
		// Verkettungen setzen 
		so2.simple = so3;
		so3.simple = so;
		so.simple = so2;
		so.sarray = new VSMessageExample[3];
		so.sarray[0] = so2;
		so.sarray[1] = null;
		so.sarray[2] = so;
		
		VSMessageExample rcv = null;
		try {
			// Verschachteltes Objekt schreiben und lesen
			out.writeObject(so);
			rcv = (VSMessageExample) in.readObject();
		} catch (Exception e) {
			e.printStackTrace();
			fail("Read/Write Object failed!");
		}
		
//		 Auswertung Objekt 1
		assertTrue("char array was not transmitted correctly", (new String(rcv.q)).equals(new String(so.q)));
		assertTrue("char field was not transmitted correctly", rcv.o == 'Y');
		assertTrue("short field was not transmitted correctly", rcv.getShort() == (short) 15);
		assertTrue("int field was not transmitted correctly", rcv.getInt() == 15000);
		assertTrue("long field was not transmitted correctly", rcv.getLong() == Long.MAX_VALUE);
		assertTrue("boolean field was not transmitted correctly", rcv.getBoolean() == true);
		assertTrue("float field was not transmitted correctly", rcv.getFloat() == (float)4.56);
		assertTrue("double field was not transmitted correctly", rcv.getDouble() == 1.234567);
		assertTrue("char field was not transmitted correctly", rcv.getChar() == 'C');
		assertTrue("byte field was not transmitted correctly", rcv.getByte() == (byte) 200);
		
		// Auswertung direkte Verknuepfung
		assertTrue("char field in referenced object was not transmitted correctly", (new String(rcv.simple.q)).equals(new String(so2.q)));
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.o == 'X');
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getShort() == (short) 17);
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getInt() == 17000);
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getLong() == (Long.MAX_VALUE - 3));
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getBoolean() == false);
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getFloat() == (float)4.76);
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getDouble() == 1.234767);
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getChar() == 'G');
		assertTrue("char field in referenced object was not transmitted correctly", rcv.simple.getByte() == (byte) 70);
		
		// Auswertung Array
		assertTrue("char field in referenced object was not transmitted correctly", rcv.sarray[0].o == 'X');
		assertTrue("referenced object null is not null after transmission", rcv.sarray[1] == null);
		assertTrue("char field in referenced object was not transmitted correctly", rcv.sarray[2].o == 'Y');
		
		// Auswertung verkettete Liste
		assertTrue("chained list was not transmitted correctly", rcv.simple.simple.simple.o == 'Y');
	
		end();
	}

	public static void main(String[] args) {
		junit.textui.TestRunner.run(VSMarshallingTest.class);	
	}
}
