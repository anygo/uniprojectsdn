package vsue.faults;

import vsue.communication.VSMessage;
import vsue.rpc.VSRemoteObjectManager;

public class VSInvokeMultipleTimes {

	public Object invokeMethod(VSMessage message) {
		return VSRemoteObjectManager.getInstance().invokeMethod(
				message.getObjectID(), message.getMethodName(),
				message.getArguments());
	}
}
