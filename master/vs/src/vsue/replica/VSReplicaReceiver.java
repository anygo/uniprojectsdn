package vsue.replica;

import java.util.HashMap;
import java.util.Map;

import org.jgroups.ExtendedReceiverAdapter;
import org.jgroups.Message;

import vsue.VSConstants;
import vsue.communication.VSMessage;
import vsue.faults.VSServerWorkerThread;
import vsue.rpc.VSRemoteObjectManager;

public class VSReplicaReceiver extends ExtendedReceiverAdapter {
	private HashMap<String, Box> map;
	private Map<String, VSServerWorkerThread> worker;

	public VSReplicaReceiver() {
		worker = new HashMap<String, VSServerWorkerThread>();
	}

	public void put(String id, VSServerWorkerThread w) {
		synchronized (worker) {
			worker.put(id, w);
		}
	}

	public Object invokeMethodMultipleTimes(VSMessage message) {
		return VSRemoteObjectManager.getInstance().invokeMethod(
				message.getObjectID(), message.getMethodName(),
				message.getArguments());
	}

	public Object invokeMethodOnce(VSMessage message) {
		Object returnValue = null;

		synchronized (map) {
			if (map.containsKey(message.getClient())
					&& map.get(message.getClient()).remoteCallID == message
							.getRemoteCallID()) {
				returnValue = map.get(message.getClient()).object;
			} else {
				returnValue = VSRemoteObjectManager.getInstance().invokeMethod(
						message.getObjectID(), message.getMethodName(),
						message.getArguments());
				map.put(message.getClient(),
						new Box(returnValue, message.getRemoteCallID()));
			}
		}

		return returnValue;
	}

	@Override
	public byte[] getState() {
		byte[] a = VSRemoteObjectManager.getInstance().getRemoteObjectStates();
		return a;
		
	}

	@Override
	public void setState(byte[] state) {
		VSRemoteObjectManager.getInstance().setRemoteObjectStates(state);
	}

	// TODO: Hier die Erkennung rein, wenn neuer Teilnehmer in die Gruppe kommt
	public void receive(Message msg) {
		VSMessage message = (VSMessage) msg.getObject();
		Object returnValue = null;
		
		if (VSConstants.RPC_SEMANTICS == VSConstants.RPC_SEMANTICS_ENUM.LAST_OF_MANY
				|| VSConstants.RPC_SEMANTICS == VSConstants.RPC_SEMANTICS_ENUM.MAYBE) {
			returnValue = invokeMethodMultipleTimes(message);
		} else if (VSConstants.RPC_SEMANTICS == VSConstants.RPC_SEMANTICS_ENUM.AT_MOST_ONCE) {
			returnValue = invokeMethodOnce(message);
		}

		synchronized (worker) {
			String key = "" + message.getRemoteCallID()
					+ message.getSequenceNumber();

			if (worker.containsKey(key)) {
				worker.get(key).setReturnValue(returnValue);
			}
		}
	}
}
