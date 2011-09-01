package vsue.faults;


import java.io.IOException;
import java.io.Serializable;

import vsue.VSConstants;
import vsue.communication.VSConnection;
import vsue.communication.VSObjectConnection;

public class VSBuggyObjectConnection extends VSObjectConnection {

	public VSBuggyObjectConnection(VSConnection con) {
		super(con);
	}
	
	public void sendObject(Serializable object) throws IOException {
		 
		int r = (int) (Math.random() * 100.0);

		try {
			System.out.println("BuggyObjectConnection - Zufallszahl r: " + r);
			if (r % 5 == 0) {
				// warten bis 5 vor 12
				System.out.println("Warten bis 5 vor 12");
				Thread.sleep(VSConstants.SEND_MULTIPLE_TIME_WAITING_TIME - 1000);
				super.sendObject(object);
			} else if (r % 3 == 0) {
				// zu lang warten
				System.out.println("zu lang warten");
				Thread.sleep(VSConstants.SEND_MULTIPLE_TIME_WAITING_TIME + 1000);
				super.sendObject(object);
			} else if (r % 7 == 0) {
				// Nachricht wird gar nicht gesendet
				System.out.println("Nachricht wird gar nicht gesendet");
			} else if (r % 4 == 0) {
				// Nachricht verfielfachen
				System.out.println("Nachricht verfielfachen"); 	
				super.sendObject(object);		// Hier ist nicht viel zu sehen, weil der Server 
				super.sendObject(object);		// nur einmal empfaengt und einmal sendet.
				super.sendObject(object);		// Danach macht er die Verbindung zu.
			} else {
				System.out.println("Sende normal");
				super.sendObject(object);
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	/*public Serializable receiveObject() throws IOException, ClassNotFoundException {
		byte[] ret = m_connection.receiveChunk();
		
		if (ret == null) return null;

		ByteArrayInputStream bais = new ByteArrayInputStream(ret);
		ObjectInputStream ois = new ObjectInputStream(bais);

		return (Serializable) ois.readObject();
	}*/

}
