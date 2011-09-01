package vsue.faults;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.rmi.Remote;
import java.rmi.RemoteException;

import vsue.communication.VSConnection;
import vsue.communication.VSMessage;
import vsue.communication.VSMessageReply;
import vsue.communication.VSObjectConnection;
import vsue.replica.VSRemoteGroupReference;
import vsue.rpc.VSRemoteObjectManager;

public abstract class VSAbstractRPCSemantics {
	private VSRemoteGroupReference groupReference;

	public VSAbstractRPCSemantics(VSRemoteGroupReference groupReference) {
		this.setRemoteGroupReference(groupReference);
	}

	public abstract Object invoke(Object proxy, Method method, Object[] args)
			throws Throwable;

	protected VSObjectConnection createConnection() {
		Socket socket = null;
		VSObjectConnection connection = null;
		boolean worked = false;
		int ct = 0;

		do {
			try {
				socket = new Socket(
						getRemoteGroupReference().getReferences()[ct].getHost(),
						getRemoteGroupReference().getReferences()[ct].getPort());
				connection = new VSObjectConnection(new VSConnection(socket));
				worked = true;
			} catch (Exception e) {
				System.out.println("Replica nicht erreichbar: " + getRemoteGroupReference().getReferences()[ct].getHost());
				
				// better safe than sorry
				worked = false;
				++ct;
				if (ct == 4) {
					e.printStackTrace();
					return null;
				}
				
			}
		} while (!worked);

		return connection;
	}

	protected VSMessageReply sendAndReceive(VSMessage clone) throws Throwable {
		VSObjectConnection connection = null;
		VSMessageReply r = null;
		
		connection = createConnection();
		
		// all replicas were not available
		if (connection == null)
		{
			throw new RemoteException("All Replicas were unavailable!");
		}
		
		System.out.println("Sending: " + clone);
		connection.sendObject(clone);
		r = (VSMessageReply) connection.receiveObject();
		System.out.println("Received reply: " + r);
		connection.close();

		return r;
	}

	protected void setRemoteGroupReference(VSRemoteGroupReference groupReference) {
		this.groupReference = groupReference;
	}

	protected VSRemoteGroupReference getRemoteGroupReference() {
		return groupReference;
	}

	// VSMessage erzeugen
	protected VSMessage generateMessage(Method method, Object[] args)
			throws UnknownHostException {

		exportRemoteArgs(args);

		// better safe than sorry
		return new VSMessage(
				getRemoteGroupReference().getReferences()[0].getObjectID(),
				method.toGenericString(), args, InetAddress.getLocalHost()
						.getCanonicalHostName()
						+ ManagementFactory.getRuntimeMXBean().getName()
						+ Thread.currentThread().getName());
	}

	// exportierbare objekte durch deren exportiertes objekt ersetzen
	protected void exportRemoteArgs(Object[] args) {
		for (int i = 0; i < args.length; ++i) {
			if (Remote.class.isAssignableFrom(args[i].getClass())) {
				if (!VSRemoteObjectManager.getInstance().isStubExported(
						(Remote) args[i])) {
					args[i] = VSRemoteObjectManager.getInstance().exportObject(
							(Remote) args[i]);
				}
			}
		}

	}

}
