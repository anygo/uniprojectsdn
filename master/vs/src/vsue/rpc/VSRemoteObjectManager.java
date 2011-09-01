package vsue.rpc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.rmi.Remote;
import java.util.HashMap;
import java.util.Map;

import vsue.VSConstants;
import vsue.replica.VSRemoteGroupReference;
import vsue.replica.VSRemoteObjectStateHandler;
import vsue.replica.VSReplicaServer;

public class VSRemoteObjectManager {
	private static VSRemoteObjectManager remoteManager;
	private static VSReplicaServer vsServer;
	private Map<Integer, Remote> exportedObjects;
	private Map<Integer, Remote> exportedObjectStubs;
	private int numberOfObjects;

	private VSRemoteObjectManager() {
		numberOfObjects = 0;
		exportedObjects = new HashMap<Integer, Remote>();
		exportedObjectStubs = new HashMap<Integer, Remote>();
	}

	public static VSRemoteObjectManager getInstance() {
		if (remoteManager == null) {
			remoteManager = new VSRemoteObjectManager();
			vsServer = new VSReplicaServer();
			vsServer.refreshState();
			System.out.println("Starting Server");
			(new Thread(vsServer)).start();
		}
		return remoteManager;
	}

	public boolean isStubExported(Remote remote) {
		return exportedObjectStubs.containsValue(remote);
	}

	public Remote exportObject(Remote object) {
		VSRemoteGroupReference groupReference = null;
		VSInvocationHandler handler = null;
		ClassLoader loader = null;
		Class<?>[] interfaceClass = null;
		Remote remote = null;

		try {
			if (exportedObjects.containsValue(object)) {
				for (Integer i : exportedObjects.keySet()) {
					if (exportedObjects.get(i) == object)
						return exportedObjectStubs.get(i);
				}
			}
			exportedObjects.put(++numberOfObjects, object);
			
			VSRemoteReference[] references = new VSRemoteReference[3];
			references[0] = new VSRemoteReference(VSConstants.REPLICA_1,
					VSConstants.VSSERVER_STARTING_PORT, numberOfObjects);
			references[1] = new VSRemoteReference(VSConstants.REPLICA_2,
					VSConstants.VSSERVER_STARTING_PORT, numberOfObjects);
			references[2] = new VSRemoteReference(VSConstants.REPLICA_3,
					VSConstants.VSSERVER_STARTING_PORT, numberOfObjects);
			groupReference = new VSRemoteGroupReference();
			groupReference.setReferences(references);

			handler = new VSInvocationHandler(groupReference);
			loader = object.getClass().getInterfaces()[0].getClassLoader();
			interfaceClass = object.getClass().getInterfaces();
			remote = (Remote) Proxy.newProxyInstance(loader, interfaceClass,
					handler);
			exportedObjectStubs.put(numberOfObjects, remote);
			
			vsServer.refreshState();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return remote;
	}

	public Object invokeMethod(int objectID, String genericMethodName,
			Object[] args) {
		Remote object = null;
		Class<?> clazz = null;

		try {
			object = exportedObjects.get(objectID);
			clazz = object.getClass();
			do {
				for (Class<?> interfaze : clazz.getInterfaces()) {
					if (!Remote.class.isAssignableFrom(interfaze))
						continue;
					for (Method m : interfaze.getMethods()) {
						if (m.toGenericString().equals(genericMethodName)) {
							try {
								return m.invoke(object, args);
							} catch (InvocationTargetException e) {
								return e.getTargetException();
							}
						}
					}
				}
			} while ((clazz = clazz.getSuperclass()) != null);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public byte[] getRemoteObjectStates() {
		ByteArrayOutputStream baos = null;
		ObjectOutputStream oos = null;

		try {
			baos = new ByteArrayOutputStream();
			oos = new ObjectOutputStream(baos);

			for (Integer i : exportedObjects.keySet()) {
				if (exportedObjects.get(i) instanceof VSRemoteObjectStateHandler) {
					oos.writeObject(i);
					oos.writeObject((((VSRemoteObjectStateHandler) exportedObjects
							.get(i)).getState()));
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		return baos.toByteArray();
	}

	public void setRemoteObjectStates(byte[] states) {
		System.err.println("setRemoteObjectStates");
		ByteArrayInputStream bais = new ByteArrayInputStream(states);
		ObjectInputStream ois;
		Integer i;
		
		try {
			ois = new ObjectInputStream(bais);
			while ((i = (Integer) ois.readObject()) != null) {
				byte[] state = (byte[]) ois.readObject();
				if (exportedObjects.get(i) instanceof VSRemoteObjectStateHandler) {
					((VSRemoteObjectStateHandler) exportedObjects.get(i))
							.setState(state);
				} 
			}
		} catch (EOFException e) {

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
