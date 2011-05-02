package vsue.tests;

import java.io.File;
import java.io.OutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.FileInputStream;

import junit.framework.TestCase;
import vsue.marshalling.*;

public class VSExternalizableTest extends TestCase {

	private OutputStream out = null;
	private InputStream in = null;
	private String tmpName;
	private VSObjectOutputStream objOut = null;
	private VSObjectInputStream objIn = null;

	public void init() {

		try {
			int id = (int) (Math.random() * 10000);
			tmpName = new String("/tmp/tmp." + id);
			out = new FileOutputStream(tmpName);
			in = new FileInputStream(tmpName);
		} catch (Exception e) {
			System.out.println("Error: file open: " + e);
			e.printStackTrace();
		}

		try {
			objOut = new VSObjectOutputStream(out);
			objIn = new VSObjectInputStream(in);
		} catch (Exception e) {
			System.out.println("Error: create streams: " + e);
			e.printStackTrace();
		}

	}

	public void end() {

		File tmp = new File(tmpName);
		tmp.delete();
		try {
			objIn.close();
			objOut.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public VSExternalizableTest(String name) {
		super(name);
	}
	
	public void testVSExternalizableExample() {

		init();
		
		char[] c = (new String("Hallo Welt")).toCharArray();
		VSExternalizableExample vse = new VSExternalizableExample(1, 2.5, c);
		VSExternalizableExample result = null;
		
		try {
			objOut.writeObject(vse);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Error writeObject");
		}

		try {
			result = (VSExternalizableExample)objIn.readObject();
		} catch (Exception e) {
			e.printStackTrace();
			fail("Error readObject");
		}

		System.out.println("Object vse: " + vse);
		System.out.println("Object result: " + result);
		assertTrue("Objects differ", (vse.toString()).equals(result.toString()));

		end();
	}

	public static void main(String argv[]) {
		junit.textui.TestRunner.run(VSExternalizableTest.class);
	}
}

