/**
 * Test zu Aufgabe 3: Proxy
 * 
 * Autor: Thomas
 * Zugehörige Dateien: VSRemoteObject.java VSRemoteObjectImpl.java
 * 
 * */

package vsue.tests;

import java.net.InetSocketAddress;

import junit.framework.TestCase;
import vsue.proxy.VSClient;
import vsue.proxy.VSRemoteObjectManager;
import vsue.proxy.VSRemoteReference;
import vsue.proxy.VSServer;

public class VSProxyTest extends TestCase {
	private int port = 20000 + (int) (Math.random() * 10000);
	private VSServer server = null;
	
	public void init () {
		// Initialisierung
		server = new VSServer();
		server.init(port);
	}
	
	public void end () {
		// Aufraeumen
	}
	
	public VSProxyTest(String name) {
		super(name);
	}
	
	public void testRemoteObjectManager() {
		// Testen ob VSRemoteObjectManager richtig funktioniert

		VSRemoteObjectManager manager = new VSRemoteObjectManager(new InetSocketAddress("localhost", 12345));
		VSRemoteObject obj = new VSRemoteObjectImpl();
		obj.setString("Test");
		manager.exportObject(obj);
		VSRemoteReference ref = manager.getRemoteReference(VSRemoteObject.class);
		assertTrue ("VSRemoteObjectManager liefert keine Referenz", ref != null);
		
		try {
			String getStringGMN = VSRemoteObject.class.getMethod("getString", (Class[]) null).toGenericString();
			assertTrue ("VSRemoteObject nicht ueber VSRemoteObjectManager zugreifbar",  ((String) manager.invokeMethod(ref.getObjectID(), getStringGMN, null)).equals("Test"));

			String concatGMN = VSRemoteObject.class.getMethod("concat", new Class[] { String.class }).toGenericString();
			assertTrue ("VSRemoteObject nicht ueber VSRemoteObjectManager zugreifbar.", ((String) manager.invokeMethod(ref.getObjectID(), concatGMN, new Object[] { "nachricht" })).equals("Testnachricht"));
			
			String setStringGMN = VSRemoteObject.class.getMethod("setString", new Class[] { String.class }).toGenericString();
			manager.invokeMethod(ref.getObjectID(), setStringGMN, new Object[] { "Test2" });
			assertTrue ("VSRemoteObject nicht ueber VSRemoteObjectManager zugreifbar.", ((String) manager.invokeMethod(ref.getObjectID(), getStringGMN, null)).equals("Test2"));
			
			String concat2GMN = VSRemoteObject.class.getMethod("concat", new Class[] { String.class, String.class }).toGenericString();
			assertTrue ("VSRemoteObject nicht ueber VSRemoteObjectManager zugreifbar.", ((String) manager.invokeMethod(ref.getObjectID(), concat2GMN, new Object[] { "pre", "post" })).equals("preTest2post"));
			
		} catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}
	
	public void testRemoteObject() {
		// Testen ob der komplette Fernaufruf funktioniert
		
		init();
		VSRemoteObject obj = new VSRemoteObjectImpl();
		obj.setString("Test");
//	System.out.println("exporting...");
		server.exportObject(obj);
		VSClient client = new VSClient();
		client.init();
//	System.out.println("performing lookup...");
		VSRemoteObject remote = (VSRemoteObject) client.lookup("localhost", port, VSRemoteObject.class);
//	System.out.println("calling function...");
		assertTrue ("VSRemoteObject Methoden nicht entfernt aufrufbar.", remote.getString().equals("Test"));
//	System.out.println("passed 1");
		assertTrue ("VSRemoteObject Methoden nicht entfernt aufrufbar.", remote.concat("nachricht").equals("Testnachricht"));
//	System.out.println("passed 2");
		remote.setString("Test2");
		assertTrue ("VSRemoteObject Methoden nicht entfernt aufrufbar.", remote.getString().equals("Test2"));
//	System.out.println("passed 3");
		assertTrue ("VSRemoteObject Methoden nicht entfernt aufrufbar.", remote.concat("pre", "post").equals("preTest2post"));

		remote.setString(null);
		assertTrue("VSRemoteObject 'null' als Parameter und/oder Rueckgabewert nicht uebertragen", (remote.getString() == null));

		end();
	}

	public static void main(String[] args) {
		// Main fuer Aufruf aus der Konsole 
		junit.textui.TestRunner.run(VSProxyTest.class);	
		System.exit(0);
	}
}
