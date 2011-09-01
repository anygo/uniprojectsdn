package vsue.faults;

import java.io.IOException;
import java.io.Serializable;
import java.rmi.Remote;
import java.util.concurrent.Semaphore;

import org.jgroups.JChannel;
import org.jgroups.Message;

import vsue.VSConstants;
import vsue.communication.VSMessage;
import vsue.communication.VSMessageReply;
import vsue.communication.VSObjectConnection;
import vsue.rpc.VSRemoteObjectManager;

public class VSServerWorkerThread implements Runnable {
	private VSObjectConnection objConnection;
	private JChannel channel;
	private Object returnValue;
	private Semaphore sem;
	private VSMessage message;

	public VSServerWorkerThread(VSObjectConnection con, JChannel channel, VSMessage message) {
		setObjConnection(con);
		setChannel(channel);
		sem = new Semaphore(0);
		this.message = message;
	}
	
	@Override
	public void run() {
		VSMessageReply reply = null;

		try {
			System.out.println("---> WorkerThread started");
			
			Message jGroupMessage = new Message(null, null, message);
			channel.send(jGroupMessage);
			
			sem.acquireUninterruptibly();
		
			// pruefen ob returnValue vom Typ Remote ist, ggf. exportieren
			if (returnValue != null && Remote.class.isAssignableFrom(returnValue.getClass())) {
				if (!VSRemoteObjectManager.getInstance().isStubExported((Remote) returnValue)) {
					returnValue = VSRemoteObjectManager.getInstance().exportObject(
							(Remote) returnValue);
				}
			}

			reply = new VSMessageReply(message.getRemoteCallID(),
					message.getSequenceNumber(), returnValue);
			
			System.out.println("Reply to " + reply);
			getObjConnection().sendObject((Serializable) reply);
			getObjConnection().close();
			System.out.println("---> Client finished");
		} catch (Exception e) {
			if (!(e instanceof IOException && e.getMessage().equals(
					VSConstants.CLOSED_MESSAGE))) {
				e.printStackTrace();
			}
		}
	}
	
	public void setReturnValue(Object returnValue) {
		this.returnValue = returnValue;
		sem.release();
	}

	public void setChannel(JChannel channel) {
		this.channel = channel;
	}

	public JChannel getChannel() {
		return channel;
	}
	
	protected void setObjConnection(VSObjectConnection objConnection) {
		this.objConnection = objConnection;
	}

	protected VSObjectConnection getObjConnection() {
		return objConnection;
	}	
}
