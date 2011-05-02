package vsue.proxy;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.lang.reflect.InvocationHandler;

public class VSInvocationHandler implements InvocationHandler {

	public static Object createProxy(VSRemoteReference ref, Class interfaceClass) {

		return Proxy.newProxyInstance(
				interfaceClass.getClassLoader(),
				new Class[] { interfaceClass },
				new VSInvocationHandler(ref)
			);

	}

	private VSRemoteReference m_ref;

	public VSInvocationHandler(VSRemoteReference ref) {

		m_ref = ref;

	}

	public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {

		// TODO 4.2: handle call-by-value vs. call-by-reference
		VSMethodObject methodCall = new VSMethodObject(m_ref.getObjectID(), method, args);

		// Send VSMethodObject to server and receive result
		VSMessage msg = new VSMessage(methodCall);
		VSCommunication vcom = new VSCommunication();
		VSConnection lookup = vcom.openConnection(m_ref.getHost(), m_ref.getPort());
		
		if(!lookup.sendMessage(msg)) {
			return null;
		}
		
		msg = lookup.receiveMessage();
		if(msg == null) {
			/*
			 * Failed to receive message
			 * 
			 * Possible reasons are:
			 * - Message was not serializable
			 * - connection lost
			 */
			System.out.println("DEBUG: invoke: failed to receive message from server");
			throw new IOException();
		}

		// Close connection to server
		vcom.closeConnection(lookup);		

		if(msg.getException() != null) {
			throw msg.getException();
		}
	
		// unwrap VSResultObject
		VSResultObject res = msg.getVSResultObject();
		if(res == null) {
			throw new Exception("Got wrong object type");
		}
		
		return res.getObject();

	}

}
