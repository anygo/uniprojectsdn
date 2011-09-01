package vsue.communication;

import java.net.Socket;

public class VSClient {

	public static void main(String[] args) throws Exception {
		String s1 = "Hallo Welt";
		Integer i1 = Integer.MIN_VALUE;
		Integer i2 = Integer.MAX_VALUE;
		
		System.out.println("Opening socket...");
		Socket sock = new Socket("faui05c", 12345);
		System.out.println("Opening VSObjectConnection...");
		VSObjectConnection con = new VSObjectConnection(new VSConnection(sock));
		
		
		String[] arr = {"abcd", "abcd", "abcd"};
		
		System.out.println("Sending array...");
		con.sendObject(arr);
		System.out.println("Receiving array...");
		/*Object obj2 = */con.receiveObject();
		
		System.out.println("Sending test string...");
		con.sendObject(s1);
		System.out.println("Receiving test string...");
		Object obj = con.receiveObject();
		if (!s1.equals((String) obj)) {
			System.out.println("Fail!");
		} else {
			System.out.println("Ok");
		}
		
		System.out.println("Sending test integer i1...");
		con.sendObject(i1);
		System.out.println("Receiving test integer...");
		obj = con.receiveObject();
		if (!i1.equals((Integer) obj)) {
			System.out.println("Fail!");
		} else {
			System.out.println("Ok");
		}
		
		System.out.println("Sending test integer i2...");
		con.sendObject(i2);
		System.out.println("Receiving test integer...");
		obj = con.receiveObject();
		if (!i2.equals((Integer) obj)) {
			System.out.println("Fail!");
		} else {
			System.out.println("Ok");
		}
		
		VSTestMessage msg = new VSTestMessage(42, "test", new Object[3]);
		System.out.println("Sending test VSMessage...");
		con.sendObject(msg);
		con.receiveObject();
		
		/*VSTestMessage null_Msg = new VSTestMessage(0, null, null);
		System.out.println("Sending null VSMessage...");
		con.sendObject(null_Msg);
		con.receiveObject();*/
		
		
		con.close();
		System.out.println("Socket closed");
	}
}
