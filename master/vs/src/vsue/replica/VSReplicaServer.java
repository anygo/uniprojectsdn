package vsue.replica;

import java.util.HashMap;

import org.jgroups.ChannelException;
import org.jgroups.JChannel;
import org.jgroups.stack.ProtocolStack;

import vsue.VSConstants;
import vsue.communication.VSMessage;
import vsue.communication.VSObjectConnection;
import vsue.communication.VSServer;
import vsue.faults.VSServerWorkerThread;
import vsue.totalorder.VSTotalOrder;

public class VSReplicaServer extends VSServer {

	private JChannel channel;

	private VSReplicaReceiver receiver;

	public VSReplicaServer() {
		super();
		try {
			channel = new JChannel();
			ProtocolStack pStack = channel.getProtocolStack();
		
			// aufgabe 5: statt new SEQUENCER()
			pStack.addProtocol(new VSTotalOrder());
		
			channel.connect(VSConstants.GROUP_NAME);
			receiver = new VSReplicaReceiver();
			channel.setReceiver(receiver);
		} catch (ChannelException e) {
			e.printStackTrace();
		}
	}

	public void refreshState() {
		try {
			System.out.println(channel.getState(null,
					VSConstants.JGROUPS_GET_STATE_TIMEOUT));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void startExecution(VSObjectConnection vsoCon,
			HashMap<String, Box> map) throws Exception {
		VSMessage message = (VSMessage) vsoCon.receiveObject();
		VSServerWorkerThread w = new VSServerWorkerThread(vsoCon, channel,
				message);
		receiver.put(
				"" + message.getRemoteCallID() + message.getSequenceNumber(), w);
		(new Thread(w)).start();
	}
}
