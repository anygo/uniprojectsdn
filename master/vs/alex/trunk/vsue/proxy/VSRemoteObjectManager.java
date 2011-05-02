package vsue.proxy;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.util.Vector;

import vsue.tests.VSRemote;

public class VSRemoteObjectManager {
	
	private InetSocketAddress m_ServerAddress;
	private Vector<Object> m_list;

	public VSRemoteObjectManager(InetSocketAddress serverAddress) {

		m_ServerAddress = serverAddress;
		m_list = new Vector<Object>();

	}

	public void exportObject(Object object) {

		if(VSRemote.class.isAssignableFrom(object.getClass())) {
				m_list.add(object);
		}

	}

	public VSRemoteReference getRemoteReference(Class interfaceClass) {
		
		for(int i = 0; i < m_list.size(); i++) {

			Class cl = m_list.get(i).getClass();
			Class[] clArray = cl.getInterfaces();
			
			for(Class c: clArray) {

				if(c == interfaceClass) {
					return new VSRemoteReference(
							m_ServerAddress.getHostName(),
							m_ServerAddress.getPort(), i);
				}	
			}

		}

		return null;

	}

	public Object invokeMethod(int objectID, String genericMethodName, Object[] args) throws Exception {

		// TODO 4.2: handle call-by-value vs. call-by-reference
		Object object = m_list.get(objectID);
		Class[] clArray = object.getClass().getInterfaces();
		
		for(Class c: clArray) {
			
			Method[] mArray = c.getMethods();
			
			for(int i = 0; i < mArray.length; i++) {
				
				if((mArray[i].toGenericString()).equals(genericMethodName)) {
					return mArray[i].invoke(object, args);
				}
				
			}
		}

		// throw RemoteException?
		throw new NoSuchMethodException();

	}

}

